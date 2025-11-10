# 标准库
import os
import random
import json
import time
from typing import List, Dict
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import concurrent.futures
from functools import lru_cache

# 第三方库
import PyPDF2
from docx import Document
import requests
from dotenv import load_dotenv

load_dotenv()

# 保持文件读取、文本拆分函数不变（无需修改）
def extract_text_from_epub(file_path: str) -> str:
    """从EPUB文件中提取文本内容"""
    text_chunks = []
    try:
        # EPUB实际上是一个ZIP文件
        with ZipFile(file_path, 'r') as z:
            # 查找content.opf文件来获取内容文档列表
            content_opf_path = None
            for file in z.namelist():
                if file.endswith('content.opf'):
                    content_opf_path = file
                    break
                    
            # 如果找不到content.opf，尝试直接查找HTML/XHTML文件
            if not content_opf_path:
                html_files = [f for f in z.namelist() if f.endswith('.html') or f.endswith('.xhtml')]
                for html_file in html_files:
                    try:
                        with z.open(html_file) as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            # 简单的HTML文本提取
                            import re
                            # 移除HTML标签
                            text = re.sub(r'<[^>]+>', ' ', content)
                            # 替换多个空白字符为单个换行
                            text = re.sub(r'\s+', ' ', text)
                            if text.strip():
                                text_chunks.append(text.strip())
                    except:
                        continue
            else:
                # 解析content.opf找到内容文档
                with z.open(content_opf_path) as f:
                    opf_content = f.read().decode('utf-8', errors='ignore')
                    
                # 创建ElementTree并获取命名空间
                root = ET.fromstring(opf_content)
                # 处理可能的命名空间
                ns = {}
                for prefix, uri in root.attrib.items():
                    if prefix.startswith('xmlns:'):
                        ns[prefix[6:]] = uri
                    elif prefix == 'xmlns':
                        ns[''] = uri
                
                # 查找所有内容文档引用
                manifest = root.find('.//{%s}manifest' % ns.get('', ''))
                spine = root.find('.//{%s}spine' % ns.get('', ''))
                
                # 获取所有item引用并排序
                item_map = {}
                if manifest is not None:
                    for item in manifest.findall('.//{%s}item' % ns.get('', '')):
                        if item.get('href') and (item.get('media-type') == 'application/xhtml+xml' or item.get('media-type') == 'text/html'):
                            item_map[item.get('id')] = item.get('href')
                
                # 按照spine顺序处理文档
                if spine is not None:
                    for itemref in spine.findall('.//{%s}itemref' % ns.get('', '')):
                        item_id = itemref.get('idref')
                        if item_id in item_map:
                            href = item_map[item_id]
                            # 获取正确的文件路径
                            base_dir = os.path.dirname(content_opf_path)
                            file_path = os.path.join(base_dir, href).replace('\\', '/')
                            
                            if file_path in z.namelist():
                                try:
                                    with z.open(file_path) as f:
                                        content = f.read().decode('utf-8', errors='ignore')
                                        # 简单的HTML文本提取
                                        import re
                                        # 移除HTML标签
                                        text = re.sub(r'<[^>]+>', ' ', content)
                                        # 替换多个空白字符为单个空格
                                        text = re.sub(r'\s+', ' ', text)
                                        if text.strip():
                                            text_chunks.append(text.strip())
                                except:
                                    continue
                
                # 如果通过spine没有找到内容，尝试所有HTML/XHTML文件
                if not text_chunks:
                    html_files = [f for f in z.namelist() if f.endswith('.html') or f.endswith('.xhtml')]
                    for html_file in html_files:
                        try:
                            with z.open(html_file) as f:
                                content = f.read().decode('utf-8', errors='ignore')
                                # 简单的HTML文本提取
                                import re
                                # 移除HTML标签
                                text = re.sub(r'<[^>]+>', ' ', content)
                                # 替换多个空白字符为单个空格
                                text = re.sub(r'\s+', ' ', text)
                                if text.strip():
                                    text_chunks.append(text.strip())
                        except:
                            continue
    except Exception as e:
        print(f"处理EPUB文件失败：{e}")
    
    # 合并所有文本块
    return "\n\n".join(text_chunks)

@lru_cache(maxsize=128)
def extract_text_from_file(file_path: str) -> str:
    file_ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if file_ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
        elif file_ext == ".epub":
            # 处理EPUB文件格式
            text = extract_text_from_epub(file_path)
        else:
            raise ValueError(f"不支持的文件格式：{file_ext}，仅支持PDF/TXT/DOCX/EPUB")
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
        return text
    except Exception as e:
        print(f"读取文件{file_path}失败：{e}")
        return ""

@lru_cache(maxsize=256)
def split_text_into_chunks(text: str, chunk_size: int = 600) -> List[str]:
    """智能分割文本为指定大小的块，支持中英文混合文本
    
    Args:
        text: 要分割的文本
        chunk_size: 文本块大小，推荐值：
            - 中文文本：500-700字符
            - 英文文本：800-1000字符
            - 混合文本：600-800字符
    
    Returns:
        分割后的文本块列表
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # 针对中英文混合文本的处理
    # 首先检查文本是否主要包含中文
    if len([c for c in text if '\u4e00' <= c <= '\u9fff']) > len(text) * 0.3:
        # 中文文本为主，使用"。"作为主要分割符，回退到"。"和"\n"
        sentences = []
        temp = ""
        for c in text:
            temp += c
            if c in ["。", "！", "？", "\n"] and len(temp) > 10:
                sentences.append(temp)
                temp = ""
        if temp:  # 处理最后一个不完整的句子
            sentences.append(temp)
    else:
        # 英文文本为主，使用标准的句号分割
        sentences = text.split(". ")
    
    # 构建文本块
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        # 检查是否需要添加到当前块或开始新块
        if current_chunk and len(current_chunk) + len(sent) + 1 <= chunk_size:
            # 添加分隔符，根据文本类型选择
            separator = "\n" if any(c in ["。", "！", "？"] for c in current_chunk) else ". "
            current_chunk += separator + sent
        else:
            # 保存当前块并开始新块
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent
    
    # 不要忘记添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

# --------------------------
# 关键修改：Prompt加入格式+风格示例
# --------------------------
def generate_from_text_chunk(
    text_chunk: str,
    api_url: str,
    api_key: str,
    model_name: str,
    difficulty: str = "easy"
) -> Dict:
    """基于真实文本片段，生成自然对话+问答（含格式示例）"""
    # 根据难度级别调整提示词
    difficulty_config = {
        "easy": "Simple conversation, memory_query is a question that can directly extract clear information from a single turn of dialogue without involving multi-turn information association or reasoning",
        "medium": "Medium complexity conversation, memory_query requires integrating relevant information from 2-3 turns of dialogue to answer",
        "hard": "Complex conversation, memory_query requires reasonable reasoning based on multi-turn dialogue information to answer, and the answer needs to be implied in the dialogue logic but not directly exposed"
    }
    
    # Use English prompt to ensure English content generation
    prompt = f"""
    Please create a multi-turn dialogue and a memory query-answer pair based on the following text content. The dialogue should be natural and revolve around the topics in the text.
    
    REQUIREMENTS:
    1. Relevance: Generated dialogue, memory query, and answer must be closely related to the provided text content and must not deviate from the text theme.
    2. Dialog history (dialog_history): Must only contain 3-8 turns, with each turn being a concise sentence without role prefixes like "User:" or "Assistant:" - only pure dialogue content. Must extract 4-6 key information points (such as numbers, dates, names, terms, core conclusions, etc.) from the provided text and distribute them across different turns to avoid information concentration.
    3. Memory query (memory_query): Must be a clear question about the dialogue content, asked in a natural way that conforms to real communication scenarios.
    4. Memory answer (memory_answer): Must be concise and complete in information, strictly derived from the dialogue history (dialog_history), and must not directly reference information from the original text that is not reflected in the dialogue, nor fabricate external content or make unwarranted expansions.
    5. The memory_query and memory_answer MUST NOT directly reference or quote text from the original input material - they must be derived solely from the generated dialogue
    6. Please ensure the output is a valid JSON object containing three fields: dialog_history, memory_query, memory_answer
    7. Generate content according to difficulty level: {difficulty_config[difficulty]}
    
    Text content:
    {text_chunk}
    """
    
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.3,  # 保持高随机性，避免示例内容干扰
            "timeout": 30  # 减少超时时间
        }
        
        # 使用会话保持连接池，减少连接建立开销
        session = requests.Session()
        with session as s:
            response = s.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()
        
        # 增强的JSON提取和验证逻辑
        try:
            # 提取响应内容
            if 'choices' in response_data and response_data['choices']:
                content = response_data['choices'][0]['message']['content'].strip()
            else:
                raise KeyError("响应中未找到有效的choices字段")
            
            # 清理Markdown格式
            if content.startswith('```'):
                content = '\n'.join([line for line in content.split('\n') 
                                   if not line.strip().startswith('```')])
            
            # 提取JSON部分
            json_start, json_end = content.find('{'), content.rfind('}')
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end+1]
            else:
                raise ValueError("无法在响应中找到有效的JSON对象")
            
            # 解析JSON
            parsed_data = json.loads(content)
            
            # 验证必要字段
            required_fields = ['dialog_history', 'memory_query', 'memory_answer']
            if not all(field in parsed_data for field in required_fields):
                missing = [f for f in required_fields if f not in parsed_data]
                raise ValueError(f"缺少必要字段：{', '.join(missing)}")
            
            # 验证字段类型
            if not isinstance(parsed_data['dialog_history'], list):
                raise TypeError("dialog_history必须是字符串数组")
            if not isinstance(parsed_data['memory_query'], str):
                raise TypeError("memory_query必须是字符串")
            if not isinstance(parsed_data['memory_answer'], str):
                raise TypeError("memory_answer必须是字符串")
                
        except (KeyError, IndexError, json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"JSON处理错误: {str(e)}")
            raise
        return parsed_data
    except Exception as e:
        print(f"生成数据失败：{e}")
        return None

# 保持数据集生成函数不变（无需修改）
def generate_single_sample(args):
    """生成单个样本的函数，用于并行处理"""
    idx, file_path, difficulties, api_url, api_key, model_name, chunk_size = args
    try:
        # 提取文本并分割
        text = extract_text_from_file(file_path)
        if not text:
            return None
            
        chunks = split_text_into_chunks(text, chunk_size)
        if not chunks:
            return None
        
        random_chunk = random.choice(chunks)
        difficulty = difficulties[idx]  # 获取当前样本的难度
        
        # 样本生成重试逻辑
        max_retries = 3  # 减少重试次数
        for retry in range(max_retries):
            try:
                sample = generate_from_text_chunk(random_chunk, api_url, api_key, model_name, difficulty)
                if sample:
                    return sample
                time.sleep(min(1 * (retry + 1), 5))  # 简化退避策略
            except Exception as e:
                # 只在最后一次重试失败时打印
                if retry == max_retries - 1:
                    print(f"生成样本 {idx+1} 失败: {str(e)}")
                time.sleep(min(1 * (retry + 1), 5))
        
        return None
    except Exception as e:
        print(f"处理样本 {idx+1} 出错: {str(e)}")
        return None

def generate_dataset_from_files(
    file_paths: List[str],
    output_file: str,
    num_samples_total: int,
    api_url: str,
    api_key: str,
    model_name: str,
    difficulty_ratios: Dict[str, float] = None,
    chunk_size: int = 600
) -> None:
    """
    从多个文件生成数据集，并保存为JSONL格式
    
    Args:
        file_paths: 文件路径列表
        output_file: 输出文件路径
        num_samples_total: 要生成的总样本数
        api_url: API请求的URL
        api_key: API密钥
        model_name: 使用的模型名称
        difficulty_ratios: 难度分布比例，默认为60%简单、25%中等、15%困难
    """
    # 如果未提供难度比例，使用默认值
    if difficulty_ratios is None:
        difficulty_ratios = {
            "easy": 0.6,  # 简单：60%
            "medium": 0.25,  # 中等：25%
            "hard": 0.15  # 困难：15%
        }
    if not file_paths:
        print("未找到有效的文件路径")
        return
    
    # 计算每种难度的样本数量
    easy_count = int(num_samples_total * difficulty_ratios["easy"])
    medium_count = int(num_samples_total * difficulty_ratios["medium"])
    hard_count = num_samples_total - easy_count - medium_count
    
    # 生成难度序列
    difficulties = (['easy'] * easy_count) + (['medium'] * medium_count) + (['hard'] * hard_count)
    random.shuffle(difficulties)  # 打乱顺序
    
    print(f"难度分布: 简单={easy_count}, 中等={medium_count}, 困难={hard_count}")
    print(f"总样本数：{num_samples_total}，使用文件数：{len(file_paths)}")
    print(f"使用文本块大小: {chunk_size} 字符")
    print(f"开始生成数据集...")
    
    # 准备并行任务参数
    tasks = []
    for idx in range(num_samples_total):
        file_path = file_paths[idx % len(file_paths)]
        tasks.append((idx, file_path, difficulties, api_url, api_key, model_name, chunk_size))
    
    # 为确保文件写入的完整性，添加异常处理
    try:
        # 先收集所有成功的样本
        successful_samples = []
        total_tasks = len(tasks)
        
        # 使用线程池并行处理
        max_workers = min(10, total_tasks)  # 根据系统调整并行度
        print(f"使用并行度: {max_workers}")
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(generate_single_sample, task): idx for idx, task in enumerate(tasks)}
            
            # 处理完成的任务
            for i, future in enumerate(concurrent.futures.as_completed(future_to_idx)):
                idx = future_to_idx[future]
                try:
                    sample = future.result()
                    if sample:
                        successful_samples.append(sample)
                        # 每100个样本打印一次进度，避免过多输出
                        if len(successful_samples) % 100 == 0 or len(successful_samples) == num_samples_total:
                            elapsed = time.time() - start_time
                            rate = len(successful_samples) / elapsed if elapsed > 0 else 0
                            print(f"已生成 {len(successful_samples)}/{num_samples_total} 样本 ({rate:.2f} 样本/秒)")
                except Exception as e:
                    print(f"任务 {idx} 异常: {str(e)}")
                
                # 每处理100个任务显示一次总体进度
                if (i + 1) % 100 == 0 or (i + 1) == total_tasks:
                    print(f"任务进度: {i + 1}/{total_tasks}")
        
        # 写入成功生成的样本
        print(f"开始写入文件，成功生成 {len(successful_samples)} 样本")
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in successful_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"数据集生成完成！成功生成 {len(successful_samples)} 个样本")
        print(f"总耗时: {time.time() - start_time:.2f} 秒")
        print(f"平均速度: {len(successful_samples) / (time.time() - start_time):.2f} 样本/秒")
    
    except Exception as e:
        print(f"数据集生成过程中发生错误：{str(e)}")
        raise
    
    print(f"数据集生成完成！保存至 {output_file}")

# 运行入口
if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()
    
    # 读取配置
    api_key = os.getenv("API_KEY") or os.getenv("LLM_API_KEY") or input("请输入API密钥：")
    config = {
        "api_url": os.getenv("API_URL", os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")),
        "api_key": api_key,
        "model_name": os.getenv("MODEL_NAME", os.getenv("LLM_MODEL_NAME", "deepseek-chat")),
        "input_dir": "data_files",  # 使用原始的数据目录名称
        "output_file": "data/memory_train.jsonl",  # 使用原始的输出文件名
        "num_samples_total": 3300,  # 使用原始的样本数量
        "chunk_size": 1000  # 文本块大小，可根据需要调整
    }
    
    if not os.path.exists(config["input_dir"]):
        os.makedirs(config["input_dir"])
        print(f"已创建输入文件夹：{config['input_dir']}，请将PDF/TXT/DOCX/EPUB文件放入其中")
    else:
        # 获取数据目录中的文件
        file_extensions = [".txt", ".pdf", ".docx", ".epub"]
        file_paths = []
        for root, dirs, files in os.walk(config["input_dir"]):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            print(f"在 {config['input_dir']} 目录中未找到文本文件")
            exit(1)
        
        # 难度配置 - 60%简单，25%中等，15%困难
        difficulty_ratios = {
            "easy": 0.6,  # 简单：60%
            "medium": 0.25,  # 中等：25%
            "hard": 0.15  # 困难：15%
        }
        
        # 生成数据集
        print(f"开始生成数据集，目标样本数：{config['num_samples_total']}")
        generate_dataset_from_files(
            file_paths=file_paths,
            output_file=config["output_file"],
            num_samples_total=config["num_samples_total"],
            api_url=config["api_url"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            difficulty_ratios=difficulty_ratios,
            chunk_size=config["chunk_size"]
        )