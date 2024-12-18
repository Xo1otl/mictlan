# ui

trainはreducer patternするべきだと思われる

分野、質問、場合、回答があって、stateの変更はreducerを経由して行われる

表示側では、stateを読んで、適切な表示を行う

## TODO

text_inputは入力がクリアされず再レンダリングのときに勝手に再入力判定になるから直す

`play.py`はstate machineで実装すべきな気がする

dispatch(event, effect)ができるstate machineを用意してやろうかなと思う
