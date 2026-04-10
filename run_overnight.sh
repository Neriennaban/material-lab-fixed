#!/usr/bin/env bash
# ============================================================
# ferro-micro — ночной автономный цикл
# Запуск: bash run_overnight.sh
# Остановка: Ctrl+C или touch STOP
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$PROJECT_DIR/logs/loop_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_FILE="$PROJECT_DIR/progress.md"
STOP_FILE="$PROJECT_DIR/STOP"
MAX_ITERATIONS=999          # практически бесконечно
SLEEP_BETWEEN=5             # пауза между итерациями (сек)

# ─── Цвета ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ─── Инициализация ──────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"
rm -f "$STOP_FILE"

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          ferro-micro  |  Autonomous Night Loop           ║"
echo "║          $(date '+%Y-%m-%d %H:%M:%S')                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  Лог: ${YELLOW}$LOG_FILE${RESET}"
echo -e "  Стоп: ${YELLOW}touch STOP${RESET} или ${YELLOW}Ctrl+C${RESET}"
echo ""

# ─── Инициализация progress.md ──────────────────────────────
if [ ! -f "$PROGRESS_FILE" ]; then
cat > "$PROGRESS_FILE" << 'EOF'
# ferro-micro — Прогресс

## Статус: 🚀 ЗАПУЩЕН

## Выполненные этапы
<!-- Claude обновляет этот список после каждого шага -->

## Текущий этап
Фаза A — начало: аудит репозитория и настройка структуры

## Файлы созданы
<!-- список созданных файлов -->

## Известные проблемы
<!-- проблемы обнаруженные в процессе -->

## Критерии завершения
- [ ] research/literature_review.md (≥15 DOI)
- [ ] constants.py
- [ ] physics/phase_diagram.py
- [ ] physics/phase_calculator.py
- [ ] physics/grain_growth.py
- [ ] physics/pearlite.py
- [ ] physics/cct_model.py
- [ ] physics/ledeburite.py
- [ ] render/voronoi_engine.py
- [ ] render/phase_renderer.py
- [ ] render/texture_generator.py
- [ ] render/etching_simulator.py
- [ ] render/compositor.py
- [ ] render/tile_renderer.py (16K)
- [ ] api/schemas.py
- [ ] api/generator.py
- [ ] api/presets.py
- [ ] api/exceptions.py
- [ ] integration/microscope_adapter.py
- [ ] legacy/ содержит старый generator.py
- [ ] tests/unit/ (≥4 файла)
- [ ] tests/integration/
- [ ] examples/generate_16k.py
- [ ] mypy --strict ✅
- [ ] ruff check ✅
EOF
echo -e "  ${GREEN}✓ progress.md создан${RESET}"
fi

# ─── Проверка: завершён ли проект ────────────────────────────
check_completion() {
    # Считаем выполненные чекбоксы в progress.md
    local done
    done=$(grep -c "\- \[x\]" "$PROGRESS_FILE" 2>/dev/null || echo "0")
    local total=23
    echo "$done/$total"
    [ "$done" -ge "$total" ]
}

# ─── Построение промпта для итерации ────────────────────────
build_prompt() {
    local iteration=$1
    cat << EOF
Ты работаешь над проектом ferro-micro (физически обоснованный генератор микрошлифов Fe-C сталей).

СНАЧАЛА: прочитай файл progress.md — там записан текущий прогресс и что уже сделано.
ЗАТЕМ: прочитай CLAUDE.md — там контекст проекта, правила кода и структура.
ЗАТЕМ: прочитай ТЗ в prompts/spec.md.

ТЕКУЩАЯ ИТЕРАЦИЯ: #$iteration

ТВОЯ ЗАДАЧА НА ЭТУ ИТЕРАЦИЮ:
1. Определи следующий незавершённый чекбокс в progress.md
2. Реализуй соответствующий модуль ПОЛНОСТЬЮ (не заглушки, не TODO)
3. Убедись что код проходит mypy и ruff для этого модуля
4. Запусти соответствующий unit-тест если существует
5. Обнови progress.md: отметь [x] выполненное, добавь созданные файлы
6. Сделай git commit: git add -A && git commit -m "feat: <модуль> — iteration $iteration"

ПРАВИЛА:
- Работай только с ОДНИМ модулем за итерацию (но реализуй его ПОЛНОСТЬЮ)
- Каждая физическая формула — комментарий с DOI или # CALIBRATION_NEEDED
- НЕ удалять legacy/ и НЕ трогать microscope.py
- Если встречаешь ошибку — исправь её в этой же итерации, не откладывай
- После git commit — ОСТАНОВИ выполнение (следующий запуск сделает скрипт)

НАЧИНАЙ с чтения progress.md!
EOF
}

# ─── Главный цикл ────────────────────────────────────────────
cd "$PROJECT_DIR"

# Инициализация git если нужно
if [ ! -d ".git" ]; then
    git init
    git add CLAUDE.md progress.md run_overnight.sh || true
    git commit -m "chore: init ferro-micro loop" || true
    echo -e "  ${GREEN}✓ git инициализирован${RESET}"
fi

ITERATION=1
START_TIME=$(date +%s)

while [ $ITERATION -le $MAX_ITERATIONS ]; do

    # Проверка файла-стопа
    if [ -f "$STOP_FILE" ]; then
        echo -e "\n${YELLOW}⏹  Файл STOP обнаружен — завершение цикла${RESET}"
        rm -f "$STOP_FILE"
        break
    fi

    # Проверка завершения проекта
    COMPLETION=$(check_completion)
    echo -e "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "  ${BOLD}Итерация #$ITERATION${RESET}  |  Завершено: ${GREEN}$COMPLETION модулей${RESET}  |  $(date '+%H:%M:%S')"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

    if check_completion 2>/dev/null; then
        echo -e "\n${GREEN}${BOLD}🎉 ПРОЕКТ ЗАВЕРШЁН! Все 23 модуля реализованы.${RESET}"
        echo -e "   Время работы: $(( ($(date +%s) - START_TIME) / 3600 ))ч $(( ( ($(date +%s) - START_TIME) % 3600 ) / 60 ))м"
        break
    fi

    # Строим промпт итерации
    PROMPT=$(build_prompt "$ITERATION")

    # Запуск Claude Code
    echo -e "  ${CYAN}▶ Запуск claude...${RESET}"
    if claude -p "$PROMPT" \
        --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
        --max-turns 80 \
        2>&1 | tee -a "$LOG_FILE"; then
        echo -e "  ${GREEN}✓ Итерация #$ITERATION завершена успешно${RESET}"
    else
        EXIT_CODE=$?
        echo -e "  ${RED}✗ Итерация #$ITERATION завершилась с кодом $EXIT_CODE${RESET}"
        echo -e "  ${YELLOW}→ Продолжаем (ошибка залогирована)${RESET}"
    fi

    ITERATION=$((ITERATION + 1))

    # Пауза между итерациями
    if [ $ITERATION -le $MAX_ITERATIONS ]; then
        echo -e "  ${YELLOW}⏸  Пауза ${SLEEP_BETWEEN}с перед следующей итерацией...${RESET}"
        sleep $SLEEP_BETWEEN
    fi

done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════════════${RESET}"
echo -e "  Итого итераций: $((ITERATION - 1))"
echo -e "  Время работы:   $(( ELAPSED / 3600 ))ч $(( (ELAPSED % 3600) / 60 ))м $(( ELAPSED % 60 ))с"
echo -e "  Лог сохранён:   $LOG_FILE"
echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════${RESET}\n"
