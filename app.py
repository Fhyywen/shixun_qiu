from flask import Flask, render_template, request, jsonify, abort
from knowledge_base.qa_system import TimeSeriesQA
from config import Config
import os
import json
from pathlib import Path
from flask import send_from_directory

app = Flask(__name__)
app.config.from_object(Config)

# 初始化问答系统
qa_system = TimeSeriesQA(Config())

# 定义financial文件夹路径
FINANCIAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/knowledge_base')

# 确保financial文件夹存在
Path(FINANCIAL_DIR).mkdir(parents=True, exist_ok=True)

# 定义允许的文件扩展名（在文件顶部添加）
ALLOWED_EXTENSIONS = ['.txt', '.md', '.rst', '.csv', '.xlsx', '.xls', '.docx','.pdf']

# Paddle 依赖安装标记文件
PADDLE_INSTALL_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.paddle_installed')

def ensure_paddle_for_pdf():
    """首次检测到 PDF 时安装所需的 Paddle 依赖（仅执行一次）。"""
    try:
        # 已安装则跳过
        if os.path.exists(PADDLE_INSTALL_FLAG):
            return

        # 为避免阻塞主进程，使用子进程依次静默安装
        import subprocess, sys
        python_exe = sys.executable

        # 安装 paddlepaddle-gpu（按用户提供的国内源与 CUDA 版本）
        subprocess.run(
            [python_exe, '-m', 'pip', 'install', 'paddlepaddle-gpu==3.2.0', '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu118/'],
            check=True
        )

        # 安装 paddleocr[all]
        subprocess.run(
            [python_exe, '-m', 'pip', 'install', 'paddleocr[all]'],
            check=True
        )

        # 记录安装完成
        with open(PADDLE_INSTALL_FLAG, 'w', encoding='utf-8') as f:
            f.write('installed')
    except Exception:
        # 不抛到上层避免影响上传流程，可在服务端日志中查看错误
        pass

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # 尝试从不同的内容类型获取问题
        if request.is_json:
            # 处理 application/json 格式
            data = request.get_json()
            question = data.get('question', '')
            knowledge_base_path = data.get('knowledge_base_path', '')  # 新增：获取知识库路径
            print("接收到的知识库路径:", knowledge_base_path)
        else:
            # 处理 application/x-www-form-urlencoded 格式
            question = request.form.get('question', '')
            knowledge_base_path = request.form.get('knowledge_base_path', '')  # 新增：获取知识库路径

        # 如果两种方式都没获取到，尝试从查询参数获取
        if not question:
            question = request.args.get('question', '')
        if not knowledge_base_path:
            knowledge_base_path = request.args.get('knowledge_base_path', '')  # 新增：从查询参数获取

        if not question:
            return jsonify({'error': '请输入问题'})

        if not knowledge_base_path:
            knowledge_base_path = Config().DATA_PATH
        else:
            # 转换路径格式（如果是Linux格式的路径，转换为Windows格式）
            knowledge_base_path = convert_path_format(knowledge_base_path)

        print(f"转换后的知识库路径: {knowledge_base_path}")

        result = qa_system.ask_question(question, knowledge_base_path)  # 传入知识库路径
        response_data = {
            'answer': result['answer'],
            'sources': result['sources'],
            'confidence': result['confidence'],
            'source_type': result.get('source_type', 'unknown'),
            'knowledge_base_used': knowledge_base_path  # 返回使用的知识库路径
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': f'处理问题时出错: {str(e)}'})

def convert_path_format(path):
    """
    转换路径格式，确保与当前操作系统兼容
    """
    # 如果是Linux格式的路径（以/开头）且在Windows系统上
    if path.startswith('/') and os.name == 'nt':
        # 移除开头的斜杠并将剩余部分转换为Windows路径
        if path.startswith('/data/'):
            # 假设 /data/ 对应 D:/python_pj/shixun_qiu/data/
            windows_path = path.replace('/data/', 'D:/python_pj/shixun_qiu/data/')
            windows_path = windows_path.replace('/', '\\')
            return windows_path
        else:
            # 通用转换：将Linux路径转换为绝对路径
            drive_letter = os.path.splitdrive(os.getcwd())[0]  # 获取当前驱动器
            windows_path = drive_letter + path.replace('/', '\\')
            return windows_path
    else:
        # 已经是Windows格式或其他情况，直接返回
        return path



@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'initialized': True})


@app.route('/build', methods=['POST'])
def build_knowledge_base():
    try:
        num_chunks = qa_system.build_knowledge_base()
        return jsonify({'message': f'知识库构建完成，共处理 {num_chunks} 个文档块'})
    except Exception as e:
        return jsonify({'error': f'构建知识库时出错: {str(e)}'})


# 文件管理相关接口
@app.route('/list-financial-files')
def list_financial_files():
    try:
        # 列出所有允许的文件类型
        files = [f for f in os.listdir(FINANCIAL_DIR)
                 if os.path.isfile(os.path.join(FINANCIAL_DIR, f))
                 and os.path.splitext(f.lower())[1] in ALLOWED_EXTENSIONS]
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': f'获取文件列表失败: {str(e)}'})


@app.route('/upload-financial-file', methods=['POST'])
def upload_financial_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'})

    files = request.files.getlist('file')  # 获取多个文件
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': '未选择文件'})

    # 获取目标文件夹路径，默认为根目录
    folder_path = request.form.get('folder_path', '')

    success_count = 0
    for file in files:
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file and file_ext in ALLOWED_EXTENSIONS:
            try:
                # 构建完整文件路径
                if folder_path:
                    target_dir = os.path.join(FINANCIAL_DIR, folder_path)
                    Path(target_dir).mkdir(parents=True, exist_ok=True)
                    file_path = os.path.join(target_dir, file.filename)
                else:
                    file_path = os.path.join(FINANCIAL_DIR, file.filename)

                if not file_path.startswith(FINANCIAL_DIR):
                    continue  # 跳过无效路径的文件

                file.save(file_path)
                success_count += 1
            except Exception as e:
                return jsonify({'error': f'文件 {file.filename} 上传失败: {str(e)}'})
        else:
            return jsonify({'error': f'文件 {file.filename} 类型不允许: {", ".join(ALLOWED_EXTENSIONS)}'})

    return jsonify({'message': f'成功上传 {success_count} 个文件', 'count': success_count})


@app.route('/delete-financial-file', methods=['POST'])
def delete_financial_file():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': '文件名不能为空'})

    file_path = os.path.join(FINANCIAL_DIR, filename)
    file_ext = os.path.splitext(filename.lower())[1]

    # 安全检查
    if not file_path.startswith(FINANCIAL_DIR) or file_ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': '无效的文件请求'})

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'message': '文件已删除'})
        else:
            return jsonify({'error': '文件不存在'})
    except Exception as e:
        return jsonify({'error': f'删除文件失败: {str(e)}'})



@app.route('/list-financial-structure')
def list_financial_structure():
    """获取知识库目录结构（包括文件和文件夹）"""
    try:
        def get_directory_structure(root_path):
            structure = []
            for entry in os.scandir(root_path):
                if entry.name.startswith('.'):  # 跳过隐藏文件/文件夹
                    continue

                item = {
                    'name': entry.name,
                    'path': os.path.relpath(entry.path, FINANCIAL_DIR),
                    'is_dir': entry.is_dir()
                }

                # 如果是目录，递归获取子目录结构
                if entry.is_dir():
                    item['children'] = get_directory_structure(entry.path)

                structure.append(item)
            return structure

        # 获取目录结构
        structure = get_directory_structure(FINANCIAL_DIR)
        return jsonify(structure)
    except Exception as e:
        return jsonify({'error': f'获取目录结构失败: {str(e)}'})


@app.route('/build-folder', methods=['POST'])
def build_folder():
    """编译指定文件夹"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path')

        if not folder_path:
            return jsonify({'error': '文件夹路径不能为空'})

        # 构建完整路径
        full_path = os.path.join(FINANCIAL_DIR, folder_path)

        # 安全检查
        if not full_path.startswith(FINANCIAL_DIR) or not os.path.isdir(full_path):
            return jsonify({'error': '无效的文件夹路径'})

        # 调用编译函数
        num_chunks = qa_system.build_knowledge_base(full_path)
        return jsonify({'message': f'文件夹编译完成，共处理 {num_chunks} 个文档块'})
    except Exception as e:
        return jsonify({'error': f'编译文件夹时出错: {str(e)}'})


# 添加一个专门的路由来切换知识库
@app.route('/switch_knowledge_base', methods=['POST'])
def switch_knowledge_base():
    try:
        if request.is_json:
            data = request.get_json()
            knowledge_base_path = data.get('knowledge_base_path', '')
        else:
            knowledge_base_path = request.form.get('knowledge_base_path', '')

        if not knowledge_base_path:
            return jsonify({'error': '请提供知识库路径'})

        success = qa_system.switch_knowledge_base(knowledge_base_path)

        if success:
            return jsonify({'message': f'已切换到知识库: {knowledge_base_path}'})
        else:
            return jsonify({'error': f'切换知识库失败: {knowledge_base_path}'})

    except Exception as e:
        return jsonify({'error': f'切换知识库时出错: {str(e)}'})

@app.route('/get_knowledge_base', methods=['GET'])
def get_knowledge_base():
    return jsonify({'knowledge_base': qa_system.get_current_knowledge_base()})

@app.route('/create-folder', methods=['POST'])
def create_folder():
    """创建新文件夹"""
    try:
        data = request.get_json()
        folder_name = data.get('folder_name')
        parent_path = data.get('parent_path', '')

        if not folder_name:
            return jsonify({'error': '文件夹名称不能为空'})

        # 构建完整路径
        if parent_path:
            full_path = os.path.join(FINANCIAL_DIR, parent_path, folder_name)
        else:
            full_path = os.path.join(FINANCIAL_DIR, folder_name)

        # 安全检查
        if not full_path.startswith(FINANCIAL_DIR):
            return jsonify({'error': '无效的文件夹路径'})

        # 检查文件夹是否已存在
        if os.path.exists(full_path):
            return jsonify({'error': '文件夹已存在'})

        # 创建文件夹
        os.makedirs(full_path, exist_ok=True)
        return jsonify({'message': '文件夹创建成功'})
    except Exception as e:
        return jsonify({'error': f'创建文件夹失败: {str(e)}'})

if __name__ == '__main__':
    Config.ensure_directories_exist()
    app.run(
        debug=Config.DEBUG,
        port=int(os.getenv('PORT', 5000)),
        host=os.getenv('HOST', '127.0.0.1')
    )