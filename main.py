from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import openai
import os
import json
from db import query_blog_posts, get_table_schema
import uvicorn
from datetime import datetime
from dotenv import load_dotenv

# Import Google Gemini
import google.generativeai as genai

load_dotenv()

app = FastAPI(
    title="SupaChat API",
    description="Natural language to Supabase analytics API (Groq + Gemini)",
    version="1.0.0"
)

# CORS Configuration
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq (via OpenAI-compatible API)
groq_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
)

# Initialize Google Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro')
else:
    gemini_model = None

# Model selection from environment (default: groq)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "groq").lower()

# System Prompt (same for both models)
# ─────────────────────────────────────────────
# SYSTEM PROMPT  (UPDATED - No aggregates)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert Supabase analytics assistant. Convert natural language queries to structured query parameters.
Respond ONLY with valid JSON in this exact format:
{
  "explanation": "A clear 2-3 sentence conversational answer describing what you found and key insights. Be specific — mention numbers, trends, or patterns visible in the data.",
  "query_params": {
    "select_columns": "column1, column2, ...",
    "filters": {"column": "value"},
    "order_by": "column_name",
    "order_desc": true,
    "limit": 100
  },
  "chart_config": {
    "type": "line" | "bar" | "pie",
    "xKey": "column_for_x_axis",
    "yKey": "column_for_y_axis",
    "title": "Human readable chart title"
  }
}

Database Schema:
Table: blog_posts
Columns:
- id (integer)
- title (text)
- topic (text)
- views (integer)
- date (date)
- engagement_score (decimal)

CRITICAL RULES:
1. NEVER use SQL aggregate functions (AVG, SUM, COUNT, MAX, MIN) in select_columns. Only use actual column names.
2. For comparisons like "compare engagement by topic", select raw columns: "topic, engagement_score, title"
3. The frontend will handle grouping/aggregation automatically from the returned data.
4. Use column names exactly as shown — no aliases, no functions.
5. For date filters use ISO format: "2024-01-01".
6. Default limit: 100. For "top N" queries set limit to N.
7. Default order: date DESC.
8. explanation must be a useful human answer with specific numbers.
9. For topic comparisons: select_columns="topic, engagement_score, views, title", order_by="engagement_score", chart with xKey="topic", yKey="engagement_score"
10. For time trends: select_columns="date, views, title", order_by="date", chart with xKey="date", yKey="views"
11. If the query is conversational (not data-related), set query_params and chart_config to empty objects {}.

Examples:
✅ GOOD: "topic, engagement_score, views" 
❌ BAD: "topic, AVG(engagement_score)"
✅ GOOD: "date, views, title"
❌ BAD: "DATE_TRUNC('day', date), SUM(views)"
"""

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None  # Allow per-request model selection

class QueryResponse(BaseModel):
    text: str
    table: List[Dict[str, Any]]
    chart: Optional[Dict[str, Any]] = None
    model_used: str  # Track which model was used

# Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "supachat-api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "supabase_url": os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
        "available_models": ["groq", "gemini"] if gemini_model else ["groq"],
        "default_model": DEFAULT_MODEL
    }

# Chat Endpoint
@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(req: QueryRequest):
    # Determine which model to use
    model_to_use = req.model.lower() if req.model else DEFAULT_MODEL
    
    if model_to_use not in ["groq", "gemini"]:
        raise HTTPException(status_code=400, detail="Invalid model. Use 'groq' or 'gemini'")
    
    if model_to_use == "gemini" and not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini is not configured. Check GEMINI_API_KEY")

    try:
        # Build the full prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser Query: {req.query}\n\nRespond with valid JSON only:"
        
        # Call the appropriate model
        if model_to_use == "groq":
            ai_response = await call_groq(full_prompt)
        else:
            ai_response = await call_gemini(full_prompt)

        # Parse AI response
        explanation = ai_response.get("explanation", "Here are the results for your query.")
        query_params = ai_response.get("query_params", {})
        chart_config = ai_response.get("chart_config", {})

        # Execute against Supabase
        table_data = query_blog_posts(
            select_columns=query_params.get("select_columns", "*"),
            filters=query_params.get("filters"),
            order_by=query_params.get("order_by", "date"),
            order_desc=query_params.get("order_desc", True),
            limit=query_params.get("limit", 100)
        )

        # Enrich explanation with row count
        row_count = len(table_data)
        if "results" not in explanation.lower() and row_count > 0:
            explanation += f" ({row_count} record{'s' if row_count != 1 else ''} returned)"

        # Build chart payload
        chart_payload = None
        if chart_config and table_data:
            chart_payload = {
                "data": table_data,
                "type": chart_config.get("type", "bar"),
                "xKey": chart_config.get("xKey", "date"),
                "yKey": chart_config.get("yKey", "views"),
                "title": chart_config.get("title", "Results"),
            }

        return QueryResponse(
            text=explanation,
            table=table_data,
            chart=chart_payload,
            model_used=model_to_use
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

async def call_groq(prompt: str) -> dict:
    """Call Groq API"""
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=700
    )
    
    raw_content = completion.choices[0].message.content
    return json.loads(raw_content)

async def call_gemini(prompt: str) -> dict:
    """Call Google Gemini API"""
    # Gemini doesn't have native JSON mode, so we prompt for it
    response = gemini_model.generate_content(
        prompt + "\n\nIMPORTANT: Respond with valid JSON only, no markdown formatting.",
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=700,
        )
    )
    
    # Extract text and clean up markdown code blocks if present
    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    if text.startswith("```"):
        text = text[3:]  # Remove ```
    if text.endswith("```"):
        text = text[:-3]  # Remove trailing ```
    
    text = text.strip()
    return json.loads(text)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to SupaChat API (Groq + Gemini)",
        "docs": "/docs",
        "health": "/health",
        "chat": "/api/chat (POST)",
        "database": "Supabase PostgreSQL",
        "models": {
            "groq": "Llama 3.3 70B (fast, free)",
            "gemini": "Gemini 1.5 Pro (Google)"
        },
        "default_model": DEFAULT_MODEL
    }

# ✅ CORRECT (fix):
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)