# Reference metallographic images for renderer calibration

Эта папка содержит научные эталонные микрофотографии, используемые для
калибровки и SSIM-валидации семейственных renderer'ов в
`core/metallography_v3/renderers/`.

## Соглашения по именованию

```
<структура>_<марка|сплав>_<увеличение>x_<источник>.png
```

Примеры:
- `martensite_lath_aisi4140_500x_asm09.png` — реечный мартенсит, AISI 4140, ×500, ASM Handbook Vol.9
- `bainite_upper_aisi4340_1000x_bhadeshia.png` — верхний бейнит, Bhadeshia «Bainite in Steels» 2001
- `ledeburite_eutectic_500x_gost.png` — эвтектический ледебурит, ГОСТ-атлас
- `widmanstatten_steel20_500x_lakhtin.png` — Видманштеттов феррит, сталь 20, Лахтин

## Источники

Принимаются изображения из:
- ASM Handbook Vol. 9 (Metallography and Microstructures)
- Vander Voort «Metallography: Principles and Practice»
- Krauss «Steels: Processing, Structure, and Performance»
- Bhadeshia «Bainite in Steels» (открытая онлайн-версия)
- ГОСТ-атласы (3443, 5639, 8233, 1778)
- ISO 945, ASTM A247/E1077

## Phase 1

Папка создана в Phase 1 мастер-плана (см. `whimsical-wandering-dawn.md`),
но заполнение эталонами — по мере реализации sub-plan'ов Phase 2-8.
На Phase 1 достаточно двух заглушечных путей в smoke-карточках
(`martensite_lath.json`, `bainite_upper.json`) — SSIM-тесты
(`tests/renderers/test_visual_snapshot_ssim.py`) автоматически
skip'аются для карточек, чьи эталоны ещё не собраны.

## Ссылки из карточек

Файлы прописаны в поле `reference_images[].path` каждой карточки
`datasets/structure_cards/*.json`. После добавления PNG в эту папку
следующий прогон `tests/renderers/test_visual_snapshot_ssim.py
::test_reference_image_paths_resolvable` перестанет skip'ать и
активирует проверку существования путей.
