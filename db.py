from supabase import create_client, Client
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Valid columns for security
VALID_COLUMNS = ["id", "title", "topic", "views", "date", "engagement_score"]

def query_blog_posts(
    select_columns: str = "*",
    filters: Optional[Dict] = None,
    order_by: str = "date",
    order_desc: bool = True,
    limit: int = 100
) -> List[Dict]:
    """
    Query blog_posts table with filters, ordering, and limits
    Sanitizes columns to prevent SQL injection and aggregate function errors
    """
    try:
        # Clean up select_columns - remove SQL functions and invalid columns
        if select_columns and select_columns != "*":
            clean_columns = []
            for col in select_columns.split(","):
                col = col.strip()
                # Skip SQL aggregate functions and invalid columns
                if any(func in col.upper() for func in ["AVG(", "SUM(", "COUNT(", "MAX(", "MIN(", "GROUP BY", "ORDER BY"]):
                    continue
                # Only allow valid column names
                if col in VALID_COLUMNS:
                    clean_columns.append(col)
            
            final_columns = ",".join(clean_columns) if clean_columns else "*"
        else:
            final_columns = "*"
        
        # Build query
        query = supabase.table("blog_posts").select(final_columns)
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                if key in VALID_COLUMNS:
                    query = query.eq(key, value)

        # Apply ordering (only allow valid columns)
        if order_by in VALID_COLUMNS:
            if order_desc:
                query = query.order(order_by, desc=True)
            else:
                query = query.order(order_by, desc=False)

        # Apply limit
        query = query.limit(limit)

        # Execute query
        result = query.execute()
        return result.data
    except Exception as e:
        raise Exception(f"Query error: {str(e)}")

def execute_raw_sql(sql: str) -> List[Dict]:
    """Execute raw SQL via Supabase RPC (SELECT only)"""
    sql = sql.strip().rstrip(";")
    if not sql.upper().startswith("SELECT"):
        raise ValueError("⛔ Security: Only SELECT queries are allowed")

    try:
        result = supabase.rpc('execute_sql', {'sql_query': sql}).execute()
        return result.data
    except Exception as e:
        raise Exception(f"Raw SQL error: {str(e)}. Use query_blog_posts() instead.")

# Compatibility alias for mcp_server.py
def execute_query(sql: str) -> List[Dict]:
    """
    Compatibility alias for mcp_server.py.
    Attempts raw SQL first, falls back to full table fetch.
    """
    try:
        return execute_raw_sql(sql)
    except Exception:
        # Fallback: return all blog posts if raw SQL isn't available
        return query_blog_posts()

def get_table_schema(table_name: str = "blog_posts") -> dict:
    """Get table schema information"""
    return {
        "blog_posts": {
            "columns": ["id", "title", "topic", "views", "date", "engagement_score"],
            "types": ["integer", "text", "text", "integer", "date", "decimal"]
        }
    }.get(table_name, {"columns": [], "types": []})