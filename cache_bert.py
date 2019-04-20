from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import json
import sys

import bert

from bert import run_classifier
from bert import optimization
from bert import tokenization

import pdb

OUTPUT_DIR = "/scratch/ovd208/nlu/my_e2e-coref/e2e-coref/bert_output_dir"


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=False)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["sequence_output"]

  #explore output_layer
  pdb.set_trace()
  #print(1/0)

  hidden_size = output_layer.shape[-1].value

  # If we're predicting, we want predicted labels and the probabiltiies.
  if is_predicting:
    return (output_layer_emb)

  return (output_layer_emb)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:


      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (output_layer_emb) = create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'emb': output_layer_emb
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


def cache_dataset(data_path, session, token_ph, len_ph, lm_emb, out_file):

  BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

  def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
      bert_module = hub.Module(BERT_MODEL_HUB)
      tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
      with tf.Session() as sess:
        vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

  tokenizer = create_tokenizer_from_hub_module()

  MAX_SEQ_LENGTH = 512
  BATCH_SIZE = 1

  run_config = tf.estimator.RunConfig(model_dir=OUTPUT_DIR)

  label_list = [1]
  model_fn = model_fn_builder(num_labels=len(label_list), learning_rate=0.001, num_train_steps=1, num_warmup_steps = 1)    

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={"batch_size": BATCH_SIZE})




  #TODO
  #update this part - take care of different BERT alignment, store a map of normal alignment to BERT alignment
 
  with open(data_path) as in_file:
    for doc_num, line in enumerate(in_file.readlines()):
      example = json.loads(line)
      sentences = example["sentences"]

      sentences = " ".join([" ".join(s) for s in sentences])

      #check sentences format - need a single string

      bert_input = bert.run_classifier.InputExample(guid=None, text_a = sentences, text_b = None, label = 1)
      

      features = bert.run_classifier.convert_examples_to_features([bert_input], label_list, MAX_SEQ_LENGTH, tokenizer)

      current_input_fn = run_classifier.input_fn_builder(
        features=features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

      pred = estimator.predict(current_input_fn)

      print(next(pred))
      #save this pred["emb"]
      pdb.set_trace()
      #print(1/0)

if __name__ == "__main__":
  #token_ph, len_ph, lm_emb = build_bert()
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    with h5py.File("bert_cache.hdf5", "w") as out_file:
      for json_filename in sys.argv[1:]:
        cache_dataset(json_filename, session, None, None, None, out_file)
