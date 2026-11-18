"""只读数据库查询工具 (所在层: L3)"""

import re
import pymysql
from langchain_core.tools import tool
from app import config


# 系统库：member 现在也能驱动 MediaAgent 生成 SQL，不允许借 SELECT 去读 MySQL 的元数据 / 账号表。
# 引用一个库只能写成 `库名.表名`，所以匹配“库名后面紧跟点”（允许反引号与空白，大小写不敏感）
_SYSTEM_SCHEMA_RE = re.compile(
    r"(?<![\w$])`?(?:information_schema|mysql|performance_schema|sys)`?\s*\.", re.IGNORECASE
)
# 制造延迟 / 占用锁的函数：几条语句就能拖垮 MySQL 连接
_DELAY_LOCK_FUNC_RE = re.compile(r"\b(?:SLEEP|BENCHMARK|GET_LOCK)\s*\(", re.IGNORECASE)
# 加锁读：SELECT 不该在只读查询里持有行锁（FOR SHARE 是 MySQL 8 里 LOCK IN SHARE MODE 的新写法）
_LOCKING_READ_RE = re.compile(r"\bFOR\s+(?:UPDATE|SHARE)\b|\bLOCK\s+IN\s+SHARE\s+MODE\b", re.IGNORECASE)
# MySQL 的版本化注释 /*!50000 ... */ 里的内容会被当作代码执行，所以扫描时只去掉注释的定界符、保留里面的文本，
# 否则 `/*!50000 SLEEP*/(5)` 能绕过 `SLEEP(` 的检查
_COMMENT_DELIMITERS_RE = re.compile(r"/\*[!+]?[0-9]*|\*/")


def _prepare_select_sql(sql: str) -> str:
    """预处理 SELECT SQL 语句：校验单条 SELECT 并确保包含 LIMIT。"""
    if not re.match(r"^\s*SELECT", sql, re.IGNORECASE):
        raise ValueError("Security Check Failed. Only SELECT statements are allowed.")
    s = sql.strip()
    while s.endswith(";"):
        s = s[:-1].strip()
    if ";" in s:
        raise ValueError("Security Check Failed. Multiple statements are not allowed.")
    # 下面的规则都在“扫描副本”上匹配（执行的仍是 s 本身）
    scan = _COMMENT_DELIMITERS_RE.sub(" ", s)
    if re.search(r"\bINTO\s+(OUTFILE|DUMPFILE)\b|\bLOAD_FILE\s*\(", scan, re.IGNORECASE):
        raise ValueError("Security Check Failed. File read/write functions are not allowed.")
    if _SYSTEM_SCHEMA_RE.search(scan):
        raise ValueError("Security Check Failed. System schemas (information_schema, mysql, performance_schema, sys) are not allowed.")
    if _DELAY_LOCK_FUNC_RE.search(scan):
        raise ValueError("Security Check Failed. SLEEP / BENCHMARK / GET_LOCK are not allowed.")
    if _LOCKING_READ_RE.search(scan):
        raise ValueError("Security Check Failed. Locking reads (FOR UPDATE / FOR SHARE / LOCK IN SHARE MODE) are not allowed.")
    if not re.search(r"\bLIMIT\b", s, re.IGNORECASE):
        s = s + "\nLIMIT 50"
    return s


@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """执行 SQL SELECT 查询并返回结果。当需要精确过滤/多条件筛选/统计计算（如女优身高、罩杯统计、发行年份过滤等）本库 MySQL 数据时调用。数据库没有题材/标签字段，题材标签请改用 av_graph_rag。"""
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
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
            read_timeout=30,
            write_timeout=30,
            init_command="SET SESSION TRANSACTION READ ONLY",
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
