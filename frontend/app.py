import sys

sys.path.append("..")

import os
import json
import traceback
from flask import Flask, request, jsonify, render_template

from data_gen import generate_pair


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/generate_sentences', methods=['POST'])
def generate_sentences():
    try:
        data = request.json
        good_grammar = data['good_grammar']
        bad_grammar = data['bad_grammar']
        config = {"sep": data['sep'], "strict_MP": data['strict_MP']}
        phrase_file = "../assets/zh_phrase.json"
        # import time
        # start = time.time()
        data = generate_pair(data['vocab'], good_grammar, bad_grammar, debug=True, phrase_file=phrase_file, **config)
        # print("Time taken:", time.time() - start)
        data.update({'success': True})

        return jsonify(data)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@app.route("/save-rules", methods=["POST"])
def save_rules():
    data = request.json
    project = data.get('project')
    print(project)
    override = data.pop("override", False)

    if not project or not data["uid"]:
        return jsonify(success=False, error="Missing project, filename")

    project_path = os.path.join("../projects", project)
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    uid, user = data["uid"], data["user"]
    file_path = f"{project_path}/{uid}_{user}.json"

    if os.path.exists(file_path) and not override:
        return jsonify({'status': 'error', 'message': 'Save name already exists. Please use a different name.'}), 400

    data.pop("vocab")
    data.pop("project")
    data["good_rule"] = data.pop("good_grammar")
    data["bad_rule"] = data.pop("bad_grammar")
    print(data)

    with open(file_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))
    
    return jsonify({"status": "success"})

@app.route("/save-vocabulary", methods=["POST"])
def save_vocabulary():
    data = request.json
    vocab = data.get('vocab')
    print(vocab)

    if not vocab:
        return jsonify(success=False, error="Missing vocabulary")

    try:
        # Specify the directory where vocabularies will be saved
        save_dir = "../assets/"
        
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Generate a filename based on the current timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zh_vocab_saved.tsv"
        
        # Combine the directory and filename
        file_path = os.path.join(save_dir, filename)

        # Write the vocabulary to the file
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(vocab)
        
        return jsonify(success=True, message=f"Vocabulary saved successfully as {filename}")
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route("/list-projects", methods=["GET"])
def list_folders():
    try:
        projects = [name for name in os.listdir("../projects") if os.path.isdir(os.path.join("../projects", name))]
        return jsonify(success=True, projects=projects)
    except Exception as e:
        return jsonify(success=False, error=str(e))


@app.route("/list-saved-files", methods=["GET"])
def list_saved_files():
    project = request.args.get("project")
    if not project:
        return jsonify(success=False, error="Folder parameter is missing")
    
    project_path = os.path.join("../projects", project)
    if not os.path.exists(project_path) or not os.path.isdir(project_path):
        return jsonify(success=False, error="Folder does not exist")

    try:
        # List all files in the specified folder
        files = sorted([name for name in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, name))])
        return jsonify(success=True, files=files)
    except Exception as e:
        return jsonify(success=False, error=str(e))
    

@app.route("/load-rules", methods=["GET"])
def load_rules():
    project = request.args.get("project")
    filename = request.args.get("filename")
    print(project, filename)
    
    if not project or not filename:
        return jsonify(success=False, error="Missing project or filename parameter")
    
    file_path = os.path.join("../projects", project, filename)
    if not os.path.exists(file_path):
        return jsonify(success=False, error="File does not exist")

    try:
        # Load rules from the specified file
        data = json.load(open(file_path, encoding='utf-8'))
        data["success"] = True
        print(data)
        return jsonify(data)
    
    except Exception as e:
        return jsonify(success=False, error=str(e))



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
