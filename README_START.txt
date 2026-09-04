Минимальная стартовая структура для Uzbek NER API.

1. Скопируйте содержимое этой папки в корень решения.
2. Положите checkpoint в model/model.safetensors.
3. Установите transformers/torch.
4. Запустите:
   python scripts/prepare_model_bundle.py --validate

После успешной проверки папка model/ станет self-contained для инференса.
Следующий шаг — service/schemas.py, service/predictor.py и service/app.py.
