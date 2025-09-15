from flask import Flask, render_template, request, jsonify, abort
from knowledge_base.qa_system import TimeSeriesQA
from config import Config
import os
import json
from pathlib import Path

app = Flask(__name__)
app.config.from_object(Config)

# 初始化问答系统
qa_system = TimeSeriesQA(Config())

# 定义financial文件夹路径
FINANCIAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/knowledge_base')

# 确保financial文件夹存在
Path(FINANCIAL_DIR).mkdir(parents=True, exist_ok=True)


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
        else:
            # 处理 application/x-www-form-urlencoded 格式
            question = request.form.get('question', '')

        # 如果两种方式都没获取到，尝试从查询参数获取
        if not question:
            question = request.args.get('question', '')

        if not question:
            return jsonify({'error': '请输入问题'})

        result = qa_system.ask_question(question)
        response_data = {
            'answer': result['answer'],
            'sources': result['sources'],
            'confidence': result['confidence'],
            'source_type': result.get('source_type', 'unknown')
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': f'处理问题时出错: {str(e)}'})

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
        # 只列出md文件
        files = [f for f in os.listdir(FINANCIAL_DIR)
                 if os.path.isfile(os.path.join(FINANCIAL_DIR, f))
                 and f.lower().endswith('.md')]
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': f'获取文件列表失败: {str(e)}'})


@app.route('/get-financial-file')
def get_financial_file():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': '文件名不能为空'})

    file_path = os.path.join(FINANCIAL_DIR, filename)

    # 安全检查，确保不会访问到financial文件夹外的文件
    if not file_path.startswith(FINANCIAL_DIR) or not filename.lower().endswith('.md'):
        return jsonify({'error': '无效的文件请求'})

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': f'读取文件失败: {str(e)}'})


@app.route('/upload-financial-file', methods=['POST'])
def upload_financial_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'})

    if file and file.filename.lower().endswith('.md'):
        try:
            file_path = os.path.join(FINANCIAL_DIR, file.filename)
            file.save(file_path)
            return jsonify({'message': '文件上传成功'})
        except Exception as e:
            return jsonify({'error': f'文件上传失败: {str(e)}'})
    else:
        return jsonify({'error': '只允许上传md文件'})


@app.route('/delete-financial-file', methods=['POST'])
def delete_financial_file():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': '文件名不能为空'})

    file_path = os.path.join(FINANCIAL_DIR, filename)

    # 安全检查
    if not file_path.startswith(FINANCIAL_DIR) or not filename.lower().endswith('.md'):
        return jsonify({'error': '无效的文件请求'})

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'message': '文件已删除'})
        else:
            return jsonify({'error': '文件不存在'})
    except Exception as e:
        return jsonify({'error': f'删除文件失败: {str(e)}'})


@app.route('/save-financial-file', methods=['POST'])
def save_financial_file():
    data = request.get_json()
    filename = data.get('filename')
    content = data.get('content')

    if not filename or content is None:
        return jsonify({'error': '文件名和内容不能为空'})

    file_path = os.path.join(FINANCIAL_DIR, filename)

    # 安全检查
    if not file_path.startswith(FINANCIAL_DIR) or not filename.lower().endswith('.md'):
        return jsonify({'error': '无效的文件请求'})

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'message': '文件保存成功'})
    except Exception as e:
        return jsonify({'error': f'保存文件失败: {str(e)}'})


if __name__ == '__main__':
    Config.ensure_directories_exist()
    app.run(
        debug=Config.DEBUG,
        port=int(os.getenv('PORT', 5000)),
        host=os.getenv('HOST', '127.0.0.1')
    )