# Базовая модель и LoRA: пять заранее заданных обращений GitHub

Вариант: all_layers. Генерация без случайного выбора, до 40 новых токенов. Версия базы: c1899de289a04d12100db370d81485cdf75e47ca.

Пять искусственных обращений зафиксированы в params.yaml до обучения. Это демонстрация подключения адаптера и формата ответа, не оценка качества на репрезентативной выборке. Итоговая тестовая часть ДЗ3 не использована.

Оба варианта получают одну и ту же системную инструкцию и шаблон из папки адаптера.

**Системная инструкция:**

> Classify the GitHub issue from its title and description. Use bug for incorrect existing behavior, feature_request for a new capability or improvement, and question for a request for usage guidance. Return only a JSON object with the key "category" and one of these three values.

## 1. Ожидаемая категория: bug

> Title: Saving a file unexpectedly removes its last line
>
> Description:
> After updating the editor, saving a text file deletes the last line even though I did not edit that line. I can reproduce this with a new file and all extensions disabled. The file should be saved without losing any content.

**Базовая модель:**

````text
{
  "category": "bug"
}
````

**С адаптером:**

````text
{"category": "bug"}
````

## 2. Ожидаемая категория: feature_request

> Title: Allow exporting search results as a CSV file
>
> Description:
> It would be useful to add an Export CSV button to the search results panel. Currently I have to copy every result manually. Please consider adding this capability in a future version.

**Базовая модель:**

````text
{
  "category": "feature_request"
}
````

**С адаптером:**

````text
{"category": "feature_request"}
````

## 3. Ожидаемая категория: question

> Title: How can I select rows matching two conditions?
>
> Description:
> I am learning pandas and have a DataFrame with age and country columns. What is the correct way to keep rows where age is greater than 18 and country is Germany? I would appreciate an example using the existing API.

**Базовая модель:**

````text
{
  "category": "question"
}
````

**С адаптером:**

````text
{"category": "question"}
````

## 4. Ожидаемая категория: bug

> Title: Grouped sum returns the wrong result for negative values
>
> Description:
> A column containing -3 and -5 in the same group produces a grouped sum of 8 instead of -8. The ordinary sum produces -8. This worked in the previous release and now gives an incorrect result.

**Базовая модель:**

````text
{
  "category": "bug"
}
````

**С адаптером:**

````text
{"category": "bug"}
````

## 5. Ожидаемая категория: feature_request

> Title: Add a configurable keyboard shortcut for switching profiles
>
> Description:
> The profile selector already works through the menu. I would like an option to assign a keyboard shortcut to switch directly to a named profile, so that this workflow needs fewer clicks.

**Базовая модель:**

````text
{
  "category": "feature_request"
}
````

**С адаптером:**

````text
{"category": "feature_request"}
````

## Что показывает эта проверка

Правильная категория вместе со строгим форматом ответа (чистый JSON без Markdown): база 5/5, адаптер 5/5.
Если убрать только обрамление блока кода Markdown, категории совпадают с ожидаемыми: база 5/5, адаптер 5/5. Это отдельная проверка содержания: нарушение формата не следует выдавать за ошибку выбора категории.
Пять примеров не доказывают статистически значимого улучшения. Изменение val loss на всех 281 отложенном примере показано в отчёте обучения.
