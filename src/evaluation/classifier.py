import numpy as np


class BinaryClassifierEvaluator(object):
    def __init__(self, model, entity_map, token_map, mention_map, session, threshold):
        self.eval_type = 'f-score'
        self.model = model
        self.session = session
        self.entity_map = entity_map
        self.token_map = token_map
        self.mention_map = mention_map
        self.threshold = threshold

    def eval(self, batches):
        model = self.model
        out_measure = 0.0
        num_examples = 0
        tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
        for batch in batches:
            seq_len_left_batch, seq_len_right_batch, left_tokens_batch, right_tokens_batch, mentions_batch, entities_batch, labels_batch = batch
            feed_dict = {model.seq_len_left: seq_len_left_batch, model.seq_len_right: seq_len_right_batch,
                         model.left_tokens: left_tokens_batch, model.right_tokens: right_tokens_batch,
                         model.mentions: mentions_batch, model.entities: entities_batch,
                         model.labels: labels_batch}
            preds = self.session.run([model.predictions], feed_dict=feed_dict)
            predicted_labels = np.squeeze(preds[0])
            actual_labels = np.squeeze(labels_batch)
            num_examples += len(actual_labels)
            tp, tn, fp, fn = self.update_confusion_matrix(predicted_labels, actual_labels, tp, tn, fp, fn)
        print "Evaluating model - with total examples: %d" % num_examples
        if self.eval_type == 'f-score':
            precision = float(tp) / (tp + fp) if (tp + fp) > 0.0 else 0.0
            recall = float(tp) / (tp + fn) if (tp + fn) > 0.0 else 0.0
            f_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0
            print "Precision : %0.3f%s" % (precision * 100, "%")
            print "Recall : %0.3f%s" % (recall * 100, "%")
            print "F-score : %0.3f%s" % (f_score * 100, "%")
            out_measure = f_score
        return out_measure

    @staticmethod
    def update_confusion_matrix(predicted_labels, actual_labels, tp, tn, fp, fn):
        if len(predicted_labels) != len(actual_labels):
            raise Exception("Dimension mismatch ")
        else:
            for i in range(len(predicted_labels)):
                if predicted_labels[i] == actual_labels[i]:
                    if predicted_labels[i]:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if predicted_labels[i]:
                        fp += 1
                    else:
                        fn += 1
        return tp, tn, fp, fn

    @staticmethod
    def print_output_score(predicted_labels, actual_labels):
        predicted_labels = np.squeeze(predicted_labels)
        actual_labels = np.squeeze(actual_labels)
        tp, tn, fp, fn = BinaryClassifierEvaluator.update_confusion_matrix(predicted_labels, actual_labels, 0, 0, 0, 0)
        precision = float(tp) / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = float(tp) / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0
        print "Precision : %0.3f%s , Recall : %0.3f%s, F1 score: %0.3f%s" % (
            precision * 100, "%", recall * 100, "%", f_score * 100, "%")
