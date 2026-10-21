"""只读数据库查询工具 (所在层: L3)"""

import re
import pymysql
from langchain_core.tools import tool
from app import config


def _prepare_select_sql(sql: str) -> str:
    """预处理 SELECT SQL 语句：校验单条 SELECT 并确保包含 LIMIT。"""
    if not re.match(r"^\s*SELECT", sql, re.IGNORECASE):
        raise ValueError("Security Check Failed. Only SELECT statements are allowed.")
    s = sql.strip()
    while s.endswith(";"):
        s = s[:-1].strip()
    if ";" in s:
        raise ValueError("Security Check Failed. Multiple statements are not allowed.")
    if not re.search(r"\bLIMIT\b", s, re.IGNORECASE):
        s = s + "\nLIMIT 50"
    return s


@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """执行 SQL SELECT 查询并返回结果。当需要精确过滤/多条件筛选/统计计算（如女优身高、罩杯统计、发行年份过滤等）本库 MySQL 数据时调用。"""
    print(f"\n[Tool Call: execute_sql] SQL: {sql_query}")
    
    try:
        sql_query = _prepare_select_sql(sql_query)
    except ValueError as e:
        return f"Error: {e}"

    connection = None
    host = config.MYSQL_HOST
    user = config.MYSQL_USER
    password = config.MYSQL_PASSWORD
    database = config.MYSQL_DATABASE
    port = int(config.MYSQL_PORT)

    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            result = cursor.fetchall()
            res_str = str(result)
            if len(res_str) > 4000:
                res_str = res_str[:4000] + "...（结果过长已截断，请加 LIMIT 或更精确的条件）"
            return res_str
    except Exception as e:
        return f"Database Error: {type(e).__name__} - {str(e)}"
    finally:
        if connection:
            connection.close()


TOOLS = [execute_sql]
