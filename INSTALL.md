# Инструкция по установке зависимостей

## Решение

### Вариант 1: Автоматическое исправление (рекомендуется)

Запустите скрипт:
```bash
python check_and_fix.py
```

### Вариант 2: Установка из requirements.txt

```bash
python -m pip install -r requirements.txt --upgrade
```

### Вариант 3: Ручная установка

```bash
# Установите TensorFlow 2.16+ (совместим с новым protobuf)
python -m pip install tensorflow>=2.16.0 --upgrade

# Обновите protobuf
python -m pip install protobuf>=4.0.0 --upgrade
```

## После исправления

Проверьте работу:
```bash
python -c "import tensorflow as tf; print('TensorFlow работает:', tf.__version__)"
```

Затем запустите:
```bash
python recognize.py
```

