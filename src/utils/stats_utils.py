from src.utils.mysql_connection import *


def get_scores(list_prediction_id, list_actual_id):
    mysql_connector = MySqConnector()
    if len(list_prediction_id) != len(list_actual_id):
        raise Exception("Something is wrong the size of lists do not match")
    elif len(list_actual_id) == 0:
        return 0.0, 0.0, 0.0
    else:
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(list_actual_id)):
            if list_prediction_id[i] == 'NA':
                fn += 1
                # print "Actual ID: %s ,Predicted ID : %s" % (list_actual_id[i], list_prediction_id[i])
                try:
                    actual_text = \
                        mysql_connector.execute_query(
                            'SELECT STR from UMLS.MRCONSO where SDUI="%s"' % list_actual_id[i])[0][0]
                    predicted_text = \
                        mysql_connector.execute_query(
                            'SELECT STR from UMLS.MRCONSO where SDUI="%s"' % list_prediction_id[i])[0][0]
                    # print "Actual Entity: %s ,Predicted Entity : %s" % (actual_text, predicted_text)
                except Exception as e:
                    pass
            elif list_prediction_id[i] in list_actual_id[i]:
                tp += 1
            else:
                fn += 1
                fp += 1
                # print "Actual ID: %s ,Predicted ID : %s" % (list_actual_id[i], list_prediction_id[i])
                try:
                    actual_text = \
                        mysql_connector.execute_query(
                            'SELECT STR from UMLS.MRCONSO where SDUI="%s"' % list_actual_id[i])[0][0]
                    predicted_text = \
                        mysql_connector.execute_query(
                            'SELECT STR from UMLS.MRCONSO where SDUI="%s"' % list_prediction_id[i])[0][0]
                    # print "Actual Entity: %s ,Predicted Entity : %s" % (actual_text, predicted_text)
                except Exception as e:
                    pass
        try:
            precision = 1.0 * tp / (tp + fp)
            recall = 1.0 * tp / (tp + fn)
            f_score = 2.0 * precision * recall / (precision + recall)
            return f_score, precision, recall
        except ZeroDivisionError:
            return 0.0, 0.0, 0.0
