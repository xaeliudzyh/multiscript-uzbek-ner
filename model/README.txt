Положите сюда обученный checkpoint под именем:

model/model.safetensors

После этого из корня проекта запустите:

python scripts/prepare_model_bundle.py

Скрипт скачает tokenizer/config исходной XLM-R модели ОДИН РАЗ
на этапе подготовки и сохранит их рядом с checkpoint.
В финальном Docker-контейнере загрузка из интернета не потребуется.
