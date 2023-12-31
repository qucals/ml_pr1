# Практическая работа #1. Создание инфраструктуры для простого проекта машинного обучения.

## Что нужно сделать:

- [x] Создать собственный репозиторий проекта машинного обучения в git и инфраструктуру для хранения артефактов в dvc, повторив шаги, описанные в лекции
- [x] Установить и настроить необходимое ПО для работы (virtualbox, putty, VSCode, ssh-server, системные linux утилиты и библиотеки python)
- [x] Установку необходимых библиотек python важно осуществлять в виртуальное окружение venv, сохранить версии библиотек в requirements.txt, который потом опубликовать в git
- [x] Построить модель, определить метрику, задать гиперпараметры, провести эксперимент, сохранить эксперимент
- [x] Произвести изменения в конвейере, сравнить результаты метрик

### Виртуальные машины

Необходимо создать три виртуальные машины:
1. для хранения данных: исходных датасетов и их измененных версий;
2. для проведения экспериментов, обучения моделей и создания программного кода системы;
3. для эксплуатации модели.

## Критерии для проверки

* Есть git-репозиторий, содержащий необходимую для повторения эксперимента информацию (git и dvc)
* В git должны быть сделаны как минимум все указанные в описании этапов решения задачи коммиты (commits)

## Последовательность действий (по лекции)

| Задача | Статус выполненности |
| :--- | :---: |
| 1. Постановка задачи | :heavy_check_mark: |
| 2. Создание и настройка базовой инфраструктуры проекта | :heavy_check_mark: |
| 2.1 Создание виртуальных машин VMware | :heavy_check_mark: |
| 2.2 Установка, настройка и администрирование Linux Ubuntu | :heavy_check_mark: |
| 2.3 Установка и настройка python и необходимых библиотек | :heavy_check_mark: |
| 3. Установка специального ПО | :heavy_check_mark: |
| 3.1 Настройка взаимодействия с git и dvc | :heavy_check_mark: |
| 3.2 Установка и настройка ПО JupyterHub, настройка работы с различными виртуальными окружениями в Jupyter Notebook | :heavy_check_mark: |
| 3.3 Установка и настройка VSCode | :heavy_check_mark: |
| 3.4 Настройка docker и docker-compose | :heavy_check_mark: |
| 4. Настройка рабочего окружения для проекта | :heavy_check_mark: |
| 4.1 Формирование рабочей структуры директориев | :heavy_check_mark: |
| 4.2 Подготовка python скриптов для отдельных этапов проекта, подключение к github, создание репозитория проекта в git. Загрузка сырых данных в хранилище данных, подготовка датасетов, предобработка, сохранение и загрузка рабочего датасета | :heavy_check_mark: |
| 4.3 Проведение первичных исследований в Jupyter Notebook. Подготовка кода на python для проведения экспериментов и обучения модели | :heavy_check_mark: |
| 4.4 Загрузка данных через dvc. Создание рабочего пространства в VSCode. Обучение модели, сохранение артефактов | :heavy_check_mark: |

## Процесс выполнения работы

### Создание виртуальных машин

![2.1](images/2.1%20VMs%20are%20created.png)

### Конфигурирование машин

![2.2](images/2.2%20VMs%20are%20configured.png)

### Настройка dvc и git

![3.1](images/3.1%20git%20and%20dvc.png)

### Исходные данные

Данные взяты с [kaggle - House Price Prediction Data](https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data/code).

Для подготовки данных к МО был написан скрипт ```scripts/data_scripts/data_preprocessing.py```. Дополнительно, в директории с данным скриптом находится Jupyter Nootebook версия.

### Этапы и конвейер

Описаны следующие этапы для dvc:
1. Подготовка данных - **data_preprocessing**;
2. Создание модели и ее обучение - **dt**;
3. Финальное тестирование модели - **evaluate**.

Скриншоты отработки этапов представлены ниже.

![Dvc params diff](images/Dvc%20params%20diff.png)

![Dvc metrics diff](images/Dvc%20metrics%20diff.png)

![Dvc exp show](images/Dvc%20exp%20show.png)

![Dvc repro](images/Dvc%20repro.png)

![Data-srv md5](images/Data-srv%20md5.png)
