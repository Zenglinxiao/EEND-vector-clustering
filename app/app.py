#! /usr/bin/env python3
# author: Linxiao ZENG
import time
import json
import os
from flask.globals import session
import yamlargparse
from flask import Flask, render_template, request, redirect, jsonify
from eend.serve import DiarizationServeModel
import soundfile as sf


def start(args):
    """Start a flask app instance."""

    st_time = time.time()
    diarization_model = DiarizationServeModel.from_args(args)
    print(f"diarization_model is ready to run inference!")
    load_time = time.time()

    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health():
        loaded = diarization_model is not None
        out = {'loaded': loaded}
        if loaded:
            out["loading_time"] = load_time - st_time
            out["summary"] = str(diarization_model._model)
        return jsonify(out)

    @app.route("/", methods=["GET", "POST"])
    def index():
        diarizations = ""
        if request.method == "POST":
            print("FORM DATA RECEIVED")

            if "file" not in request.files:
                return redirect(request.url)

            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)

            gold_clusters = request.form["num_speaker"]
            if not gold_clusters:
                gold_clusters = -1
                print("Infer #speaker in audio")
            else:
                gold_clusters = int(gold_clusters)
                print(f"#speaker in audio Given: {gold_clusters}")

            threshold = request.form["threshold"]
            if not threshold:
                threshold = args.threshold
                print(f"Use default threshold: {threshold}")
            else:
                threshold = float(threshold)
                print(f"Use provided threshold: {threshold}")

            if file:
                wav, rate = sf.read(file)
                print(f"loaded wav, rate {rate}")
                session_id, _ = os.path.splitext(os.path.basename(file.filename))
                rttm_list, timing = diarization_model.diarize_wav(
                    wav, rate,
                    threshold=threshold,
                    median=args.median,
                    num_clusters=gold_clusters,
                    session=session_id,
                )
                diarizations = json.dumps(rttm_list)
                print("Diarization Done!")
        return render_template('index.html', diarizations=diarizations)

    app.run(host=args.ip, port=args.port, debug=args.debug)


def _get_parser():
    parser = yamlargparse.ArgumentParser(
        description="EEND-vector-clustering demo"
    )
    parser.add_argument(
        '-c', '--config', help='config file path',
        action=yamlargparse.ActionConfigFile
    )
    DiarizationServeModel.add_args(parser)
    # audio_file receive through HTTP request
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default="8000")
    parser.add_argument("--debug", "-d", action="store_true")
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    start(args)


if __name__ == "__main__":
    main()
