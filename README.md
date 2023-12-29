# Практическая работа #1. Создание инфраструктуры для простого проекта машинного обучения.

## Что нужно сделать:

- [x] Создать собственный репозиторий проекта машинного обучения в git и инфраструктуру для хранения артефактов в dvc, повторив шаги, описанные в лекции
- [ ] Установить и настроить необходимое ПО для работы (virtualbox, putty, VSCode, ssh-server, системные linux утилиты и библиотеки python)
- [ ] Установку необходимых библиотек python важно осуществлять в виртуальное окружение venv, сохранить версии библиотек в requirements.txt, который потом опубликовать в git
- [ ] Построить модель, определить метрику, задать гиперпараметры, провести эксперимент, сохранить эксперимент
- [ ] Произвести изменения в конвейере, сравнить результаты метрик

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
| 3. Установка специального ПО | |
| 3.1 Настройка взаимодействия с git и dvc | :heavy_check_mark: |
| 3.2 Установка и настройка ПО JupyterHub, настройка работы с различными виртуальными окружениями в Jupyter Notebook | :heavy_check_mark: |
| 3.3 Установка и настройка VSCode | :heavy_check_mark: |
| 3.4 Настройка docker и docker-compose | |
| 4. Настройка рабочего окружения для проекта | |
| 4.1 Формирование рабочей структуры директориев | |
| 4.2 Подготовка python скриптов для отдельных этапов проекта, подключение к github, создание репозитория проекта в git. Загрузка сырых данных в хранилище данных, подготовка датасетов, предобработка, сохранение и загрузка рабочего датасета | |
| 4.3 Проведение первичных исследований в Jupyter Notebook. Подготовка кода на python для проведения экспериментов и обучения модели | |
| 4.4 Загрузка данных через dvc. Создание рабочего пространства в VSCode. Обучение модели, сохранение артефактов | |

## Проблемы

- [X] Сетевой интерфейс ВМ не запускается (возможно, стоит заменить образ ОС для ВМ)