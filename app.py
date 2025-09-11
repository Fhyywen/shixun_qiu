from flask import Flask, render_template, request, jsonify
from knowledge_base.qa_system import TimeSeriesQA
from config import config
import os

app = Flask(__name__)
app.config.from_object(config)

# 初始化问答系统
qa_system = TimeSeriesQA(
    data_dir=config.DATA_DIR,
    vector_store_path=config.VECTOR_STORE_PATH,
    embedding_model=config.EMBEDDING_MODEL,
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question', '')
    if not question:
        return jsonify({'error': '请输入问题'})

    try:
        k = int(request.form.get('k', config.SEARCH_RESULTS_COUNT))
        result = qa_system.ask(question, k=k)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'处理问题时出错: {str(e)}'})


@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({'status': 'healthy', 'initialized': qa_system.initialized})


if __name__ == '__main__':
    # 确保必要的目录存在
    config.ensure_directories_exist()

    # 初始化知识库
    qa_system.initialize()

    app.run(
        debug=config.DEBUG,
        port=int(os.getenv('PORT', 5000)),
        host=os.getenv('HOST', '127.0.0.1')
    )