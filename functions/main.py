import os
from http import HTTPStatus
from flask import jsonify

# LangChainの主要コンポーネントをインポート
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from firebase_admin import initialize_app
from firebase_functions.options import set_global_options, CorsOptions
from flask import Flask, jsonify
from firebase_functions import params, https_fn

# Flaskの定義 (jsonifyを使うために残します)
app = Flask(__name__)

# --- Firebase Admin SDKの初期化 ---
initialize_app()

# 関数のグローバルオプションを設定（例：最大インスタンス数）
set_global_options(max_instances=10)

# 固定 CORS ヘッダーの定義
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',  # すべてのオリジンを許可
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '3600',  # プリフライト結果をキャッシュする秒数
}

# 1. APIキーを環境変数から取得し設定
# SecretParam オブジェクトは関数の外で定義する (変更なし)
OPENAI_API_KEY = params.SecretParam('OPENAI_API_KEY') # ★ SecretParam を利用

# --- ヘルパー関数 ---

def create_response(response):
    """
    レスポンスオブジェクトに CORS ヘッダーを適用する。
    Flask Response または https_fn.Response に対応。
    """
    # Note: 'Content-Type' は jsonify が既に設定しているため、ここでは CORS 関連のみを更新
    response.headers.update(CORS_HEADERS)
    return response


# Cloud FunctionsのHTTPトリガーを定義
@https_fn.on_request(secrets=["OPENAI_API_KEY"])  # ★ cors=CorsOptions(...) の引数を削除
def rag_api_handler(request: https_fn.Request) -> tuple[https_fn.Response, int] | https_fn.Response:
    # 1. OPTIONS (プリフライトリクエスト) のハンドリング
    if request.method == 'OPTIONS':
        return create_response(
            https_fn.Response('', status=HTTPStatus.NO_CONTENT)
        )

    # ★ 2. APIキーが存在しない場合のチェックと強制終了
    openai_api_key = OPENAI_API_KEY.value
    print(f"APIキー：{openai_api_key}")
    if not openai_api_key:
        error_response = jsonify({'error': 'OpenAI API Key (OPENAI_API_KEY) not set in environment.'})
        # ★ エラー応答も必ず create_response を通し、CORSヘッダーを付与する
        return create_response(error_response), HTTPStatus.INTERNAL_SERVER_ERROR

    # 3. リクエストから質問（query）を取得
    request_json = request.get_json(silent=True)
    user_query = request_json.get('prompt', 'Firebase FunctionsのRAGについて教えてください。')  # 'prompt'を使用

    # --- RAGの「事前準備」ステップ ---

    # 4. ナレッジベースの定義
    knowledge_text = """
    【RAGシステム運用ルール】
名前:三好 智(みよし あきら)
性別:男
年齢:28歳
性格:怒らず優しく、楽しく、真面目に。

拝見して頂きありがとうございます。
マンツーマンで依頼に向けて親身になって頑張ります！
みなさま、よろしくお願いします！

まったり楽しく一緒にゴールに向けて頑張りましょう！
٩(๑>▽<๑)۶

【フォートナイト分野】
2019 フォートナイト講師活動開始
2022 有名チームに所属(NEXUS)
2024 生徒がランクマッチで世界一位に！(生徒のYoutube登録者11万人↑)
・有名プロゲーマーを倒したり、プロゲーマーとYoutube共演などなど...

【IT分野】
〈経歴〉
2015 中企業の塾講師 (大学中退)
2017 フリーランスエンジニア
2019 零細企業のフルスタックエンジニア
2022 ステップアップ転職、大手企業のフルスタックエンジニア
2024 大手ITグループの内部部署異動
2025 同じ会社内のチーム内 異動 アーキテクト専攻へ切り替え予定

【芸術分野】
・映像制作
・作曲、作詞
・AI生成可能

【プログラミング言語・フレームワーク】
C# 経験年数 : 6年
C#.NET 経験年数 : 6年
Google Apps Script 経験年数 : 2年
HTML 経験年数 : 6年
Java 経験年数 : 2年
JavaScript 経験年数 : 6年
PL/SQL 経験年数 : 1年未満
SQL 経験年数 : 6年
TypeScript 経験年数 : 2年
ASP.NET 経験年数 : 1年
Node.js 経験年数 : 2年
React 経験年数 : 2年
Unity 経験年数 : 2年
Vue.js 経験年数 : 2年
Amazon Web Services 経験年数 : 1年
Microsoft SQL Server 経験年数 : 6年
Oracle Database 経験年数 : 1年
Git 経験年数 : 6年
GitLab 経験年数 : 1年
GitHub 経験年数 : 6年

【納品経験】
・人材系大手企業システム
・教育系大手企業システム
・物流系大手企業システム
    """

    # 5. データの前処理とベクトル化
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_text(knowledge_text)

    # テキストをベクトル化し、メモリ内のベクトルストアに保存（PoC向け）
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = DocArrayInMemorySearch.from_texts(docs, embeddings)

    # --- RAGの「実行」ステップ ---

    # 6. LLMの準備
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        api_key=openai_api_key
    )

    # 7. プロンプトテンプレートの定義
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "あなたは企業のナレッジベースBotです。提供されたコンテキストのみに基づいて、正確かつ簡潔に回答してください。"),
            ("human", "コンテキスト: {context}\n質問: {input}"),
        ]
    )

    # 8. RAGチェーンの構築
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 9. 実行と回答の取得
    response_txt = retrieval_chain.invoke({"input": user_query})
    responce_data = {
        'query': user_query,
        'answer': response_txt['answer'],
        # 参照元ドキュメント（今回はコード内のナレッジ）
        'source_documents_count': len(response_txt['context']),
        'status': 'success'
    }

    # 10. 結果の整形と返却
    success_response = jsonify(responce_data)
    #終了
    print(f"回答：{response_txt['answer']}")
    print(f"応答終了")
    # create_response を適用し、ステータスコードと共に返す
    return create_response(success_response), HTTPStatus.OK