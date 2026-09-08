import json
import pymysql

# 1. MySQL 连接配置（请根据你的实际情况修改）
DB_CONFIG = {
    "host": "localhost",
    "port": 8964,
    "user": "root",          # 你的 MySQL 用户名
    "password": "8964",  # 你的 MySQL 密码
    "database": "av_db",     # 你的数据库名
    "charset": "utf8mb4"
}

def init_mysql_tables():
    """初始化 MySQL 表结构"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            # 创建演员表（包含你计划收集的4个核心字段）
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS actresses (
                name VARCHAR(100) PRIMARY KEY,
                birth_year INT,
                cup_size VARCHAR(10),
                height INT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            
            # 创建作品表（从 work.jsonl 中洗出）
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS works (
                code VARCHAR(50) PRIMARY KEY,
                title VARCHAR(500),
                year INT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            
            # 创建作品-演员关联表（多对多桥梁表）
            # 注意：这里只对 work_code 建立了外键约束。
            # 不给 actress_name 建立硬外键约束，是为了防止在 actresses 表没收录某些小众女优时导致作品关联整行报错。
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS work_actress (
                work_code VARCHAR(50),
                actress_name VARCHAR(100),
                PRIMARY KEY (work_code, actress_name),
                FOREIGN KEY (work_code) REFERENCES works(code) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
        conn.commit()
        print("✅ [MySQL] 三张核心基础表结构初始化/检查成功！")
    finally:
        conn.close()

def import_actresses(file_path):
    """读取 actresses.jsonl 并批量导入 MySQL"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            # 使用 INSERT IGNORE 保证即使重复运行脚本也不会因为主键冲突报错
            sql = """
            INSERT IGNORE INTO actresses (name, birth_year, cup_size, height)
            VALUES (%s, %s, %s, %s)
            """
            batch_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    batch_data.append((
                        row.get("name"),
                        row.get("birth_year"),
                        row.get("cup_size"),
                        row.get("height")
                    ))
            
            if batch_data:
                cursor.executemany(sql, batch_data)
                conn.commit()
                print(f"✅ [MySQL] 成功导入 {cursor.rowcount} 条演员基础数据！")
    except Exception as e:
        print(f"❌ 演员数据导入失败: {e}")
    finally:
        conn.close()

def import_works_and_relations(file_path):
    """自动解析 work.jsonl(原1.jsonl)，并洗出数据分别导入 works 表和 work_actress 关联表"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            sql_work = "INSERT IGNORE INTO works (code, title, year) VALUES (%s, %s, %s)"
            sql_relation = "INSERT IGNORE INTO work_actress (work_code, actress_name) VALUES (%s, %s)"
            
            works_batch = []
            relations_batch = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    code = row.get("code")
                    if not code: continue
                    
                    # 1. 抽取作品信息
                    works_batch.append((
                        code,
                        row.get("title"),
                        row.get("year") # 如果1.jsonl里没有year，MySQL会自动存为 NULL
                    ))
                    
                    # 2. 抽取多对多关联关系
                    actresses = row.get("actresses", [])
                    for actress in actresses:
                        if actress and actress.strip(): # 确保名字非空
                            relations_batch.append((code, actress.strip()))
            
            # 批量安全写入 works 表
            if works_batch:
                cursor.executemany(sql_work, works_batch)
                conn.commit()
                print(f"✅ [MySQL] 成功从作品源清洗并导入 {len(works_batch)} 条作品数据！")
                
            # 批量安全写入 work_actress 关联表
            if relations_batch:
                cursor.executemany(sql_relation, relations_batch)
                conn.commit()
                print(f"✅ [MySQL] 成功建立并导入 {len(relations_batch)} 条作品-演员关联映射关系！")
    except Exception as e:
        print(f"❌ 作品及关联数据导入失败: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # 执行一键迁移
    init_mysql_tables()
    
    # 💡 替换成你实际存放 jsonl 文件的路径
    import_actresses(r"D:\code\VisualStudioCode\AI\project\GNK48-Agent\QQbot\data\GraphRAG\actresses.jsonl")
    import_works_and_relations(r"D:\code\VisualStudioCode\AI\project\GNK48-Agent\QQbot\data\GraphRAG\work.jsonl")    
    print("\n🎉 恭喜，所有结构化数据已成功洗入 MySQL，可以开始进行 Text-to-SQL 的接入了！")