# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    if not target_text: return int(len(predicted_text) > 0)
    return editdistance.eval(target_text, predicted_text) / len(target_text)

def calc_wer(target_text: str, predicted_text: str) -> float:
    if not target_text: return int(len(predicted_text) > 0)
    target_splitted, predict_splitted = target_text.split(' '), predicted_text.split(' ')
    return editdistance.eval(target_splitted, predict_splitted) / len(target_splitted)
