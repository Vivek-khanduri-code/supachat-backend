from fastmcp import FastMCP
from db import execute_query, get_table_schema
import json

mcp = FastMCP("SupaChat MCP Server")

@mcp.tool()
def run_sql(query: str) -> str:
    """
    Execute a PostgreSQL SELECT query and return results as JSON
    
    Args:
        query: SQL SELECT query
        
    Returns:
        JSON string of query results
    """
    try:
        rows = execute_query(query)
        return json.dumps(rows, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def get_schema(table_name: str = "blog_posts") -> str:
    """
    Get database schema information
    
    Args:
        table_name: Name of the table (default: blog_posts)
        
    Returns:
        Schema information as string
    """
    try:
        return get_table_schema(table_name)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def list_tables() -> str:
    """List all available tables in the database"""
    query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """
    try:
        tables = execute_query(query)
        return ", ".join([t['table_name'] for t in tables])
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Run MCP server with stdio transport
    mcp.run(transport="stdio")