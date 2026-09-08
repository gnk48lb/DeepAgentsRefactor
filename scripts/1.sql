-- 1. 创建你的 av_db 数据库
CREATE DATABASE IF NOT EXISTS av_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 2. 查看当前所有的数据库，检查 av_db 是否创建成功
SHOW DATABASES;

-- 查出库里所有身高在 165cm 以上、且演过 2020 年以后作品的 G 罩杯及以上女优和作品编号
SELECT 
    a.name AS '演员姓名', 
    a.height AS '身高', 
    a.cup_size AS '罩杯', 
    w.code AS '作品番号', 
    w.title AS '作品标题', 
    w.year AS '发行年份'
FROM av_db.actresses a
JOIN av_db.work_actress wa ON a.name = wa.actress_name
JOIN av_db.works w ON wa.work_code = w.code
WHERE a.height >= 165 
  AND a.cup_size >= 'G' 
  AND w.year >= 2020
LIMIT 10;
