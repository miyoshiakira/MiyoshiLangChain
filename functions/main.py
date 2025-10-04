import os
from http import HTTPStatus
from flask import jsonify

# LangChainの主要コンポーネントをインポート
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Cloud FunctionsのHTTPトリガーを定義
# ユーザーの質問を受け付け、RAGを実行し、回答を返します
def rag_api_handler(request):
    """
    RAGシステムを実装したHTTP Function。
    ユーザーの質問に基づき、内部ナレッジを参照して回答を生成します。
    """

    # 1. APIキーを環境変数から取得し設定
    # 認証がない場合やキーがない場合はエラーを返す
    if 'OPENAI_API_KEY' not in os.environ:
        return jsonify({
            'error': 'OpenAI API Key not set.'
        }), HTTPStatus.INTERNAL_SERVER_ERROR

    # 2. リクエストから質問（query）を取得
    request_json = request.get_json(silent=True)
    user_query = request_json.get('query', 'Firebase FunctionsのRAGについて教えてください。')

    # --- RAGの「事前準備」ステップ ---
    
    # 3. ナレッジベースの定義 (今回はコード内に直接記述)
    # 本来は社内手順書やフロー設計書の内容をここに入れます。
    knowledge_text = """
    【社内RAGシステム運用ルール】
    1. システム名: KnowledgeBot V1.0
    2. 開発環境: Google Cloud Functions (Python) と LangChain
    3. データソース: 社内規程書（2024年4月版）、業務フロー設計書（最新）
    4. 特徴: RAG技術により、ハルシネーション（誤情報生成）を最小限に抑えます。
    5. 担当部署: AI推進室
    6. 連絡先: support@example.com
    """

    # 4. データの前処理とベクトル化
    # メモリ内検索用にテキストをチャンク（断片）に分割
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_text(knowledge_text)

    # テキストをベクトル化し、メモリ内のベクトルストアに保存（PoC向け）
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = DocArrayInMemorySearch.from_texts(docs, embeddings)

    # --- RAGの「実行」ステップ ---

    # 5. LLMの準備
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    # 6. プロンプトテンプレートの定義
    # LLMに「何をすべきか」を指示します
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは企業のナレッジベースBotです。提供されたコンテキストのみに基づいて、正確かつ簡潔に回答してください。"),
            ("human", "コンテキスト: {context}\n質問: {input}"),
        ]
    )

    # 7. RAGチェーンの構築
    # 検索（Retrieval）と生成（Generation）を結合
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 8. 実行と回答の取得
    response = retrieval_chain.invoke({"input": user_query})

    # 9. 結果の整形と返却
    return jsonify({
        'query': user_query,
        'answer': response['answer'],
        # 参照元ドキュメント（今回はコード内のナレッジ）
        'source_documents_count': len(response['context']),
        'status': 'success'
    }), HTTPStatus.OK