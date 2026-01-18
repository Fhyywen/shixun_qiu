import logging
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, abort

from knowledge_base.knowledge_base_analyzer import KnowledgeBaseAnalyzer
from knowledge_base.qa_system import TimeSeriesQA
from config import Config
import os
import json
from pathlib import Path
from flask import send_from_directory
from flask import Response, stream_with_context

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
    """
    功能：首次检测到PDF文件时，自动安装PaddlePaddle相关依赖（仅执行一次）
    参数：无
    返回：无
    异常：捕获安装过程中的所有异常，记录日志但不抛出（避免阻塞主流程）
    """
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
    """
       功能：渲染应用首页
       参数：无
       返回：首页HTML模板字符串
       异常：无（Flask框架会自动处理模板渲染异常）
       """
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    """
        功能：处理用户的问答请求，根据指定知识库路径回答用户问题
        参数：
            入参来源：POST请求的JSON体、表单或URL查询参数
            question: 用户提出的问题字符串（必传）
            knowledge_base_path: 知识库文件存储路径（可选，默认使用配置文件中的路径）
        返回：JSON格式响应，包含回答内容、来源文档、置信度、来源类型、使用的知识库路径
        异常：捕获所有处理过程中的异常，返回包含错误信息的JSON响应
        """
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


@app.route('/ask_stream', methods=['POST'])
def ask_question_stream():
    """
    功能：流式返回问答结果（基于SSE服务器发送事件），支持多轮对话和会话生命周期管理
    参数：
        入参来源：POST请求的JSON体或表单
        question: 用户提出的问题字符串（必传）
        knowledge_base_path: 知识库文件存储路径（可选，默认使用配置文件中的路径）
        session_id: 会话ID（可选，用于关联多轮对话上下文）
        user_id: 用户ID（可选，默认值'anonymous'，用于区分不同用户）
        new_session: 是否创建新会话（可选，布尔值，默认False）
    返回：SSE格式的流式响应，包含逐段返回的回答内容；参数错误返回400，系统异常返回500
    异常：捕获所有处理过程中的异常，记录日志并返回包含错误信息的JSON响应
    """
    try:
        if request.is_json:
            data = request.get_json()
            question = data.get('question', '')
            knowledge_base_path = data.get('knowledge_base_path', '')
            session_id = data.get('session_id', '')  # 新增：会话ID
            user_id = data.get('user_id', 'anonymous')  # 新增：用户ID
            new_session = data.get('new_session', False)  # 新增：是否创建新会话
        else:
            question = request.form.get('question', '')
            knowledge_base_path = request.form.get('knowledge_base_path', '')
            session_id = request.form.get('session_id', '')  # 新增：会话ID
            user_id = request.form.get('user_id', 'anonymous')  # 新增：用户ID
            new_session = request.form.get('new_session', False, type=bool)  # 新增：是否创建新会话

        if not question:
            return jsonify({'error': '请输入问题'}), 400

        # 处理知识库路径
        if not knowledge_base_path:
            knowledge_base_path = Config().DATA_PATH
        else:
            knowledge_base_path = convert_path_format(knowledge_base_path)

        # 会话管理逻辑
        if new_session or not session_id:
            # 创建新会话
            session_id = qa_system.create_session(user_id=user_id, knowledge_base_path=knowledge_base_path)
            print(f"创建新会话: {session_id}")
        else:
            # 验证现有会话
            try:
                history = qa_system.get_conversation_history(session_id)
                print(f"使用现有会话: {session_id}, 历史消息数: {len(history)}")
            except Exception as e:
                print(f"会话验证失败, 创建新会话: {e}")
                session_id = qa_system.create_session(user_id=user_id, knowledge_base_path=knowledge_base_path)

        headers = {
            'Content-Type': 'text/event-stream; charset=utf-8',
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*',  # 根据实际情况调整CORS
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }

        # 流式响应
        return Response(
            stream_with_context(qa_system.stream_answer_sse(
                question=question,
                knowledge_base_path=knowledge_base_path,
                session_id=session_id
            )),
            headers=headers,
            mimetype='text/event-stream'
        )

    except Exception as e:
        logging.error(f"流式问答接口错误: {str(e)}")
        return jsonify({'error': f'处理问题时出错: {str(e)}'}), 500


@app.route('/sessions', methods=['GET', 'POST', 'DELETE'])
def manage_sessions():
    """
    功能：会话全生命周期管理接口，支持获取用户会话列表、创建新会话、关闭指定会话
    参数：
        通用参数：
            user_id: 用户ID（可选，默认值'anonymous'，来源：URL查询参数或JSON请求体）
        GET方法：无额外参数，根据user_id获取会话列表
        POST方法：
            knowledge_base_path: 知识库文件存储路径（可选，默认使用配置文件路径）
            title: 会话标题（可选，默认值'新对话'）
        DELETE方法：
            session_id: 要关闭的会话ID（必传，来源：JSON请求体）
    返回：
        GET：JSON响应，包含success状态、sessions会话列表、count会话数量
        POST：JSON响应，包含success状态、session_id新会话ID、message提示信息
        DELETE：JSON响应，包含success状态、message提示信息；缺少session_id返回400
    异常：捕获所有处理过程中的异常，记录日志并返回包含错误信息的JSON响应（状态码500）
    """
    try:
        user_id = request.args.get('user_id', 'anonymous') or request.json.get('user_id', 'anonymous')

        if request.method == 'GET':
            # 获取用户会话列表
            sessions = qa_system.get_user_sessions(user_id=user_id)
            return jsonify({
                'success': True,
                'sessions': sessions,
                'count': len(sessions)
            })

        elif request.method == 'POST':
            # 创建新会话
            data = request.get_json() or {}
            knowledge_base_path = data.get('knowledge_base_path', '')
            title = data.get('title', '新对话')

            if knowledge_base_path:
                knowledge_base_path = convert_path_format(knowledge_base_path)

            session_id = qa_system.create_session(
                user_id=user_id,
                knowledge_base_path=knowledge_base_path,
                title=title
            )

            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': '会话创建成功'
            })

        elif request.method == 'DELETE':
            # 关闭会话
            data = request.get_json() or {}
            session_id = data.get('session_id')

            if not session_id:
                return jsonify({'error': '缺少session_id参数'}), 400

            qa_system.close_session(session_id)
            return jsonify({
                'success': True,
                'message': '会话已关闭'
            })

    except Exception as e:
        logging.error(f"会话管理错误: {str(e)}")
        return jsonify({'error': f'会话管理失败: {str(e)}'}), 500


@app.route('/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    """
    功能：获取指定会话ID的消息历史记录，支持限制返回消息数量
    参数：
        路径参数：
            session_id: 会话唯一标识（必传，从URL路径中获取）
        URL查询参数：
            limit: 返回消息的最大数量（可选，默认值100，需为整数）
    返回：JSON响应，包含success状态、session_id、messages消息列表、count消息总数
    异常：捕获所有处理过程中的异常，记录日志并返回包含错误信息的JSON响应（状态码500）
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        messages = qa_system.get_session_messages(session_id, limit)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'messages': messages,
            'count': len(messages)
        })
    except Exception as e:
        logging.error(f"获取会话消息错误: {str(e)}")
        return jsonify({'error': f'获取消息失败: {str(e)}'}), 500


@app.route('/sessions/<session_id>/title', methods=['PUT'])
def update_session_title(session_id):
    """
    功能：更新指定会话ID的标题
    参数：
        路径参数：
            session_id: 要更新标题的会话唯一标识（必传，从URL路径中获取）
        请求体参数（JSON格式）：
            title: 新的会话标题（必传，不能为空字符串）
    返回：
        成功：JSON响应，包含success状态、提示信息、会话ID、新标题
        失败：标题为空返回400；系统异常返回500，包含错误信息
    异常：捕获所有处理过程中的异常，记录日志并返回包含错误信息的JSON响应（状态码500）
    """
    try:
        data = request.get_json() or {}
        new_title = data.get('title', '').strip()

        if not new_title:
            return jsonify({'error': '标题不能为空'}), 400

        qa_system.db_manager.update_session_title(session_id, new_title)

        return jsonify({
            'success': True,
            'message': '标题更新成功',
            'session_id': session_id,
            'new_title': new_title
        })
    except Exception as e:
        logging.error(f"更新会话标题错误: {str(e)}")
        return jsonify({'error': f'更新标题失败: {str(e)}'}), 500



def convert_path_format(path):
    """
    功能：转换文件路径格式，将Linux风格路径适配为Windows系统兼容的路径格式
    参数：
        path: 待转换的路径字符串（可能是Linux格式/Windows格式）
    返回：适配当前操作系统的路径字符串（Windows系统返回\\分隔的路径，其他系统返回原路径）
    异常：无（仅做字符串处理，不涉及IO或外部调用，无抛出异常）
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
    """
        功能：服务健康状态检查接口，用于监控系统检测服务是否正常运行
        参数：无（GET请求，无需传入任何参数）
        返回：JSON格式响应，包含服务状态（status）和初始化状态（initialized）
        异常：无（接口仅返回固定响应，无业务逻辑和异常处理）
        """
    return jsonify({'status': 'healthy', 'initialized': True})


@app.route('/build', methods=['POST'])
def build_knowledge_base():
    """
        功能：触发知识库全量构建，解析知识库目录下的文档并生成可检索的文档块
        参数：无（POST请求，无需传入任何参数，使用配置文件中指定的知识库路径）
        返回：
            成功：JSON响应，包含构建完成提示和处理的文档块数量
            失败：JSON响应，包含构建过程中的错误信息（状态码默认200，仅返回error字段）
        异常：捕获知识库构建过程中的所有异常，返回包含错误信息的JSON响应
        """
    try:
        num_chunks = qa_system.build_knowledge_base()
        return jsonify({'message': f'知识库构建完成，共处理 {num_chunks} 个文档块'})
    except Exception as e:
        return jsonify({'error': f'构建知识库时出错: {str(e)}'})


# 文件管理相关接口
@app.route('/list-financial-files')
def list_financial_files():
    """
        功能：列出知识库目录下所有允许类型的文件（财务相关文档）
        参数：无（GET请求，无需传入任何参数）
        返回：
            成功：JSON格式的文件名称列表（仅包含允许扩展名的文件）
            失败：JSON响应，包含获取文件列表失败的错误信息
        异常：捕获文件目录读取、遍历过程中的所有异常，返回包含错误信息的JSON响应
        """
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
    """
        功能：上传一个或多个财务相关文件到知识库目录，支持指定子文件夹存储
        参数：
            表单参数：
                file: 待上传的文件（支持多文件，必传）
                folder_path: 目标子文件夹路径（可选，默认存储到知识库根目录）
        返回：
            成功：JSON响应，包含上传成功提示和成功上传的文件数量
            失败：
                - 无文件部分/未选择文件：返回对应错误信息
                - 文件类型不允许：返回不允许的文件名和支持的扩展名列表
                - 文件保存失败：返回具体文件名和失败原因
        异常：捕获文件保存过程中的异常，返回对应错误信息
        """
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
    """
        功能：删除知识库目录下指定的财务文件，包含多重安全校验防止误删/越权删除
        参数：
            请求体参数（JSON格式）：
                filename: 要删除的文件名（必传，包含扩展名）
        返回：
            成功：JSON响应，提示“文件已删除”
            失败：
                - 文件名为空：返回“文件名不能为空”
                - 文件路径无效/扩展名不允许：返回“无效的文件请求”
                - 文件不存在：返回“文件不存在”
                - 删除失败：返回具体的错误原因
        异常：捕获文件删除过程中的所有异常（如权限不足、文件被占用等），返回错误信息
        """
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
    """
    功能：递归获取知识库目录的完整结构（包含文件和子文件夹），跳过隐藏文件/文件夹
    参数：无（GET请求，无需传入任何参数）
    返回：
        成功：JSON格式的目录结构列表，每个条目包含名称、相对路径、是否为目录、子节点（目录才有）
        失败：JSON响应，包含获取目录结构失败的错误信息
    异常：捕获目录遍历、递归过程中的所有异常（如权限不足、目录不存在等），返回错误信息
    """
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
    """
       功能：编译知识库中指定文件夹内的所有文档，生成可检索的文档块（仅编译指定文件夹，非全量）
       参数：
           请求体参数（JSON格式）：
               folder_path: 待编译的文件夹路径（相对于知识库根目录，必传）
       返回：
           成功：JSON响应，包含编译完成提示和处理的文档块数量
           失败：
               - 文件夹路径为空：返回“文件夹路径不能为空”
               - 路径无效/非目录：返回“无效的文件夹路径”
               - 文件编码错误：返回UTF-8编码提示
               - 其他异常：返回具体的编译错误原因
       异常：
           - UnicodeDecodeError：专门捕获文件编码错误，返回友好提示
           - 通用Exception：捕获编译过程中的其他异常（如路径不存在、权限不足等）
       """
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

        # 调用编译函数，传递编码处理参数
        num_chunks = qa_system.build_knowledge_base(full_path)
        return jsonify({'message': f'文件夹编译完成，共处理 {num_chunks} 个文档块'})

    except UnicodeDecodeError as e:
        # 专门处理编码错误
        return jsonify({'error': f'文件编码错误: {str(e)}。请确保文件使用 UTF-8 编码'})
    except Exception as e:
        return jsonify({'error': f'编译文件夹时出错: {str(e)}'})


# 添加一个专门的路由来切换知识库
@app.route('/switch_knowledge_base', methods=['POST'])
def switch_knowledge_base():
    """
        功能：切换问答系统当前使用的知识库路径，使后续问答请求使用新的知识库
        参数：
            入参来源：POST请求的JSON体或表单
            knowledge_base_path: 目标知识库的路径字符串（必传）
        返回：
            成功：JSON响应，提示已切换到指定知识库路径
            失败：
                - 未提供知识库路径：返回“请提供知识库路径”
                - 切换操作失败：返回“切换知识库失败”+具体路径
                - 系统异常：返回具体的错误原因
        异常：捕获切换知识库过程中的所有异常（如路径不存在、权限不足、系统内部错误等），返回错误信息
        """
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
    """
       功能：获取问答系统当前正在使用的知识库路径
       参数：无（GET请求，无需传入任何参数）
       返回：JSON格式响应，包含当前使用的知识库路径（字段名：knowledge_base）
       异常：无（仅调用只读方法返回当前状态，无业务逻辑和异常处理）
       """
    return jsonify({'knowledge_base': qa_system.get_current_knowledge_base()})

@app.route('/create-folder', methods=['POST'])
def create_folder():
    """
        功能：在知识库目录下创建新文件夹，支持指定父文件夹路径
        参数：
            请求体参数（JSON格式）：
                folder_name: 新文件夹名称（必传）
                parent_path: 父文件夹路径（可选，相对于知识库根目录，默认创建在根目录）
        返回：
            成功：JSON响应，提示“文件夹创建成功”
            失败：
                - 文件夹名称为空：返回“文件夹名称不能为空”
                - 路径无效（超出知识库范围）：返回“无效的文件夹路径”
                - 文件夹已存在：返回“文件夹已存在”
                - 创建失败：返回具体的错误原因
        异常：捕获文件夹创建过程中的所有异常（如权限不足、路径非法等），返回错误信息
        """
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


# 添加模板上传接口
@app.route('/upload-template', methods=['POST'])
def upload_template():
    """
        功能：上传问答系统的回答模板文件，替换系统默认模板并动态更新配置
        参数：
            表单参数：
                template_file: 待上传的模板文件（必传）
        返回：
            成功：JSON响应，包含上传成功提示、文件保存路径、原文件名、模板类型
            失败：
                - 未选择文件/文件名为空：返回“未选择文件”
                - 文件类型不支持：返回允许的模板文件类型列表
                - 处理失败：返回具体的错误原因
        异常：捕获文件保存、目录创建、配置更新过程中的所有异常，返回错误信息
        """
    if 'template_file' not in request.files:
        return jsonify({'error': '未选择文件'})

    file = request.files['template_file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'})

    # 获取文件扩展名
    file_ext = os.path.splitext(file.filename.lower())[1]

    # 验证文件类型
    if file_ext not in Config.ALLOWED_TEMPLATE_EXTENSIONS:
        return jsonify({
            'error': f'不支持的文件类型，允许的类型: {", ".join(Config.ALLOWED_TEMPLATE_EXTENSIONS)}'
        })

    try:
        config = Config()
        # 保留原始文件扩展名
        template_filename = f"answer_template{file_ext}"
        template_path = os.path.join(config.DATA_PATH, "knowledge_base", template_filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        file.save(template_path)

        # 动态更新配置
        Config.set_answer_template(template_path)

        return jsonify({
            'message': f'模板文件上传成功，已保存至: {template_path}',
            'filename': file.filename,
            'template_type': file_ext
        })
    except Exception as e:
        return jsonify({'error': f'处理模板文件时出错: {str(e)}'})

if __name__ == '__main__':
    Config.ensure_directories_exist()
    app.run(
        debug=Config.DEBUG,
        port=int(os.getenv('PORT', 5000)),
        host=os.getenv('HOST', '127.0.0.1')
    )