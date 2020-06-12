#!/usr/bin/env python

import argparse
import os
import sys
import uuid

from glob import iglob
from os.path import dirname
from pathlib import Path
from typing import Tuple, List
sys.path.append(os.path.join(dirname(__file__), "src"))

from dpu_utils.utils import load_jsonl_gz
from ptgnn.implementations.typilus.graph2class import Graph2Class

from annotationutils import (
    annotate_line,
    find_annotation_line,
    group_suggestions,
    annotation_rewrite,
)
from changeutils import get_changed_files
from graph_generator.extract_graphs import extract_graphs

import warnings
warnings.filterwarnings("ignore")

# copy, as entyrpoint does not have main(), so import will trigger the execution
# from entrypoint import TypeSuggestion
class TypeSuggestion:
    def __init__(
        self,
        filepath: str,
        name: str,
        file_location: Tuple[int, int],
        suggestion: str,
        symbol_kind: str,
        confidence: float,
        annotation_lineno: int = 0,
        is_disagreement: bool = False,
    ):
        self.filepath = filepath
        self.name = name
        self.file_location = file_location
        self.suggestion = suggestion
        self.symbol_kind = symbol_kind
        self.confidence = confidence
        self.annotation_lineno = annotation_lineno
        self.is_disagreement = is_disagreement

    def __repr__(self) -> str:
        return (
            f"Suggestion@{self.filepath}:{self.file_location} "
            f"Symbol Name: `{self.name}` Suggestion `{self.suggestion}` "
            f"Confidence: {self.confidence:.2%}"
        )



parser = argparse.ArgumentParser(description='Inference from the pretained model using https://github.com/typilus/typilus')
parser.add_argument('--model', dest="model_path", required=True, help='path to the pretrained model in .pkl.gz format')
parser.add_argument('--repo', dest="repo_path", required=True, help='path to source code repository to analyzer')
parser.add_argument('--file', dest="file_path", required=True, help='suggest type only for a given file (must be under --repo)')
parser.add_argument('-v', dest="debug", action="store_true", default=False, help='verbose debug output')
# parser.add_argument('-', dest="diff_stdin", action="store_true", default=False, help="suggest types only for the changed files (read diff from stdin)")

# Usage:
# wget https://github.com/typilus/typilus-action/releases/download/v0.1/typilus20200507.pkl.gz
# ./typilus.py --model typilus20200507.pkl.gz --repo . --file entrypoint.py

# TODO(bzz):
# ./typilus.py --model typilus20200507.pkl.gz --repo .
# ./typilus.py --model typilus20200507.pkl.gz --repo . - < git diff master^

def main():
    args = parser.parse_args()
    debug = args.debug
    model_path = args.model_path
    repo_path = args.repo_path
    out_dir = os.path.join("graph", str(uuid.uuid4()))
    print(f"Intermediate output is saved under '{out_dir}'")

    # if args.file_path:
    changed_files = {args.file_path[len(repo_path) :]: set()}
    # else:
    #     #TODO list all files under "path" by default

    # if args.diff_stdin:
    #     # diff = <read diff from stdin>
    #     changed_files = get_changed_files(diff)

    if len(changed_files) == 0:
      print("No relevant changes found.")
      return

    Path(out_dir).mkdir(parents=True)
    typing_rules_path = os.path.join(dirname(__file__), "src", "metadata", "typingRules.json")
    assert Path(typing_rules_path).exists()
    extract_graphs(
        repo_path, typing_rules_path, files_to_extract=set(changed_files), target_folder=out_dir,
    )

    ## the rest is exactly the same as entrypoint.py
    def data_iter():
        for datafile_path in iglob(os.path.join(out_dir, "*.jsonl.gz")):
            print(f"\nLooking into {datafile_path}...")
            for graph in load_jsonl_gz(datafile_path):
                yield graph

#    model_path = os.getenv("MODEL_PATH", "/usr/src/model.pkl.gz")
    model, nn = Graph2Class.restore_model(model_path, "cpu")

    type_suggestions: List[TypeSuggestion] = []
    for graph, predictions in model.predict(data_iter(), nn, "cpu"):
        # predictions has the type: Dict[int, Tuple[str, float]]
        filepath = graph["filename"]

        if debug:
            print("Predictions:", predictions)
            print("SuperNodes:", graph["supernodes"])

        for supernode_idx, (predicted_type, predicted_prob) in predictions.items():
            supernode_data = graph["supernodes"][str(supernode_idx)]
            if supernode_data["type"] == "variable":
                continue  # Do not suggest annotations on variables for now.
            lineno, colno = supernode_data["location"]
            suggestion = TypeSuggestion(
                filepath,
                supernode_data["name"],
                (lineno, colno),
                annotation_rewrite(predicted_type),
                supernode_data["type"],
                predicted_prob,
                is_disagreement=supernode_data["annotation"] != "??"
                and supernode_data["annotation"] != predicted_type,
            )

            print("\t", suggestion)

            if lineno not in changed_files[filepath]:
                continue
            elif suggestion.name == "%UNK%":
                continue

            if (
                supernode_data["annotation"] == "??"
                and suggestion.confidence > suggestion_confidence_threshold
            ):
                type_suggestions.append(suggestion)
            elif (
                suggestion.is_disagreement
                # and suggestion.confidence > diagreement_confidence_threshold
            ):
                pass  # TODO: Disabled for now: type_suggestions.append(suggestion)

    print(f"Done, {len(type_suggestions)} suggestions found.")


if __name__ == "__main__":
    main()