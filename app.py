from flask import Flask, render_template, request, jsonify
from knowledge_base.qa_system import TimeSeriesQA
from config import Config
import os

app = Flask(__name__)
app.config.from_object(Config)

# 初始化问答系统 - 使用正确的初始化方式
qa_system = TimeSeriesQA(Config())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question', '')
    if not question:
        return jsonify({'error': '请输入问题'})

    try:
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
    """健康检查端点"""
    return jsonify({'status': 'healthy', 'initialized': True})  # 简化健康检查

@app.route('/build', methods=['POST'])
def build_knowledge_base():
    """构建知识库端点"""
    try:
        num_chunks = qa_system.build_knowledge_base()
        return jsonify({'message': f'知识库构建完成，共处理 {num_chunks} 个文档块'})
    except Exception as e:
        return jsonify({'error': f'构建知识库时出错: {str(e)}'})

if __name__ == '__main__':
    # 确保必要的目录存在
    Config.ensure_directories_exist()

    # 初始化知识库（可选，可以在Web界面中手动触发）
    # qa_system.build_knowledge_base()

    app.run(
        debug=Config.DEBUG,
        port=int(os.getenv('PORT', 5000)),
        host=os.getenv('HOST', '127.0.0.1')
    )