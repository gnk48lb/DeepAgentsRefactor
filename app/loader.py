import os
import fitz  # PyMuPDF
from PIL import Image
import base64
from io import BytesIO
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from . import models
import hashlib

def get_file_sha256(byte_data: bytes) -> str:
    return hashlib.sha256(byte_data).hexdigest()

def compress_image_to_base64(image_path: str, max_size=(1024, 1024), quality=80) -> str:
    """压缩图片并转换为 base64"""
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=quality)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error compressing image {image_path}: {e}")
        return ""

SOURCE_DIR = "data/source"
PIPELINE_IMAGES_DIR = "data/pipeline/images"
PIPELINE_SUMMARIES_DIR = "data/pipeline/summaries"

def process_pdfs_and_images(source_dir=SOURCE_DIR, is_incremental=False):
    """扫描目录下的 PDF 和图片，提取页面并调用 VLM 生成小作文缓存"""
    import shutil
    os.makedirs(PIPELINE_IMAGES_DIR, exist_ok=True)
    os.makedirs(PIPELINE_SUMMARIES_DIR, exist_ok=True)
    
    active_hashes = set()
    newly_created_hashes = set()
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_base, ext = os.path.splitext(file)
            ext = ext.lower()
            
            if ext == ".pdf":
                try:
                    pdf_doc = fitz.open(file_path)
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc.load_page(page_num)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_bytes = pix.tobytes("png")
                        img_hash = get_file_sha256(img_bytes)
                        
                        img_name = f"{img_hash}.png"
                        img_path = os.path.join(PIPELINE_IMAGES_DIR, img_name)
                        summary_path = os.path.join(PIPELINE_SUMMARIES_DIR, f"{img_hash}.md")
                        
                        is_new = not os.path.exists(summary_path)
                        if is_new:
                            newly_created_hashes.add(img_hash)
                        active_hashes.add(img_hash)
                        
                        if not os.path.exists(img_path):
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            print(f"Extracted PDF page to {img_path}")
                except Exception as e:
                    print(f"Error processing PDF {file_path}: {e}")
            elif ext in [".png", ".jpg", ".jpeg"]:
                with open(file_path, "rb") as f:
                    img_bytes = f.read()
                img_hash = get_file_sha256(img_bytes)
                
                img_path = os.path.join(PIPELINE_IMAGES_DIR, f"{img_hash}{ext}")
                summary_path = os.path.join(PIPELINE_SUMMARIES_DIR, f"{img_hash}.md")
                
                is_new = not os.path.exists(summary_path)
                if is_new:
                    newly_created_hashes.add(img_hash)
                active_hashes.add(img_hash)
                
                if not os.path.exists(img_path):
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    print(f"Copied image to {img_path}")

    # VLM 扫描 PIPELINE_IMAGES_DIR 中活跃的图片
    for root, _, files in os.walk(PIPELINE_IMAGES_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            file_base, ext = os.path.splitext(file)
            ext = ext.lower()
            
            if ext in [".png", ".jpg", ".jpeg"]:
                if file_base not in active_hashes:
                    continue
                
                summary_path = os.path.join(PIPELINE_SUMMARIES_DIR, f"{file_base}.md")
                if os.path.exists(summary_path):
                    continue
                
                print(f"Generating VLM summary for {file_path}...")
                base64_img = compress_image_to_base64(file_path)
                if not base64_img:
                    continue
                
                prompt = (
                    "请极其详尽地描述这张图片。这是一份用于知识库检索的‘小作文’。\n"
                    "如果图片是图表，请提取所有数据点、趋势、坐标轴含义。\n"
                    "如果图片包含文本，请原样提取所有文本内容。\n"
                    "如果是示意图或流程图，请详细描述各个节点和它们之间的关系。\n"
                    "请使用 Markdown 格式组织内容（可使用标题采用、列表、表格等）。"
                )
                
                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                )
                
                try:
                    res = models.vlm.invoke([msg])
                    content_res = res.content
                    rel_img_path = os.path.relpath(file_path, "data")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(f"<!-- image_path: {rel_img_path} -->\n\n")
                        f.write(content_res)
                    print(f"✅ Saved summary to {summary_path}")
                    newly_created_hashes.add(file_base)
                except Exception as e:
                    print(f"❌ Failed to generate summary for {file_path}: {e}")
                    
    return newly_created_hashes if is_incremental else active_hashes

def load_and_split_markdown(data_dir="data/source", is_incremental=False):
    """
    递归读取目录下的所有 markdown 文件，并根据标题进行切分
    """
    active_hashes = process_pdfs_and_images(data_dir, is_incremental=is_incremental)

    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = []
    
    # 构造待处理的文件列表，避免扫描非本次增量涉及的 VLM 摘要
    files_to_process = []
    
    # 1. 递归扫描 data_dir 下的 Markdown 文件
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".md"):
                    files_to_process.append((os.path.join(root, file), False))
                    
    # 2. 仅扫描本批次活跃图片所生成的 VLM 摘要
    for img_hash in active_hashes:
        summary_path = os.path.join(PIPELINE_SUMMARIES_DIR, f"{img_hash}.md")
        if os.path.exists(summary_path):
            files_to_process.append((summary_path, True))
        
    for file_path, is_summary in files_to_process:
        rel_path = os.path.relpath(file_path, "data")
        source_id = rel_path.replace("\\", "_").replace("/", "_").replace(".", "_")
        
        content_type = "image_description" if is_summary else "text"
        image_path = ""
        
        try:
            with open(file_path, "r", encoding="utf-8") as f_md:
                content_md = f_md.read()
                
            if is_summary:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                found_image = False
                for img_ext in [".png", ".jpg", ".jpeg"]:
                    potential_img = os.path.join(PIPELINE_IMAGES_DIR, f"{base_name}{img_ext}")
                    if os.path.exists(potential_img):
                        image_path = os.path.relpath(potential_img, "data")
                        found_image = True
                        break
                
                if not found_image:
                    print(f"⚠️ [Loader] 找不到摘要 {rel_path} 对应的原图，将作为普通文本处理。")
                    content_type = "text"
                
            md_header_splits = markdown_splitter.split_text(content_md)
            level_ids = {1: None, 2: None, 3: None, 4: None}
            
            chunk_counter = 0
            for header_split in md_header_splits:
                current_level = 0
                breadcrumb_parts = []
                for i in range(1, 5):
                    h_key = f"header_{i}"
                    if h_key in header_split.metadata:
                        current_level = i
                        h_val = header_split.metadata[h_key]
                        breadcrumb_parts.append(f"{'#' * i} {h_val}")
                
                breadcrumb = " > ".join(breadcrumb_parts)
                header_split.page_content = f"{breadcrumb}\n{header_split.page_content}"
                
                parent_id = None
                if current_level > 1:
                    parent_id = level_ids.get(current_level - 1)
                
                splits = text_splitter.split_documents([header_split])
                
                for i, split in enumerate(splits):
                    logical_id = f"{source_id}_{chunk_counter:03d}"
                    
                    if i == 0 and current_level > 0:
                        level_ids[current_level] = logical_id
                        for low_level in range(current_level + 1, 5):
                            level_ids[low_level] = None
                    
                    split.metadata.update({
                        "chunk_id": logical_id,
                        "source_id": source_id,
                        "parent_id": parent_id or "",
                        "source_path": rel_path,
                        "image_path": image_path,
                        "content_type": content_type
                    })
                    docs.append(split)
                    chunk_counter += 1
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            import traceback
            traceback.print_exc()
                        
    return docs

if __name__ == "__main__":
    docs = load_and_split_markdown()
    print(f"Loaded {len(docs)} document chunks.")
    if docs:
        print("Sample metadata:", docs[0].metadata)
        print("Sample content:", docs[0].page_content[:100], "...")
