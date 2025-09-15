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
    # 同时支持表单和JSON格式的请求
    if request.is_json:
        data = request.get_json()
        question = data.get('question', '')
    else:
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
        # 检查知识库目录是否有文件
        knowledge_base_path = qa_system.config.KNOWLEDGE_BASE_PATH
        if not os.path.exists(knowledge_base_path) or not any(os.scandir(knowledge_base_path)):
            return jsonify({
                'success': False,
                'message': f'知识库目录为空或不存在: {knowledge_base_path}',
                'directory': knowledge_base_path
            })

        num_chunks = qa_system.build_knowledge_base()
        if num_chunks > 0:
            return jsonify({
                'success': True,
                'message': f'知识库构建完成，共处理 {num_chunks} 个文档块',
                'collection': qa_system.config.COLLECTION_NAME,
                'chroma_db_path': qa_system.config.CHROMA_DB_PATH
            })
        else:
            return jsonify({
                'success': False,
                'message': '知识库构建失败，未处理任何文档。请检查日志获取详细信息。'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'构建知识库时出错: {str(e)}'
        })


@app.route('/stats', methods=['GET'])
def get_stats():
    """获取知识库统计信息"""
    try:
        count = qa_system.collection.count()
        return jsonify({
            'success': True,
            'collection': qa_system.config.COLLECTION_NAME,
            'document_count': count,
            'chroma_db_path': qa_system.config.CHROMA_DB_PATH
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取统计信息时出错: {str(e)}'
        })

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