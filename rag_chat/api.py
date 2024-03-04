from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chat.main import chat_engine

app = Flask(__name__)
CORS(app)


# TODO: handle memory for each call.

@app.route('/chat', methods=['POST'])
def chat():
    print("hola")
    user_message = request.json['message']
    response = chat_engine.chat(user_message)
    return jsonify(response.response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)