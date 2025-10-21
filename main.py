import re
import io
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="NL → Plot (RU)", page_icon="📊", layout="wide")

# ===================== Utilities =====================

def load_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")))

def safe_eval_expr(expr: str):
    allowed_ops = (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE)
    allowed_unary = (ast.USub, ast.UAdd, ast.Not)
    try:
        node = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError(f"Некорректное выражение фильтра: {e}")
    class Checker(ast.NodeVisitor):
        def visit_Attribute(self, n): raise ValueError("Атрибуты запрещены в фильтре.")
        def visit_Call(self, n): raise ValueError("Вызовы функций запрещены в фильтре.")
        def visit_Compare(self, n):
            for op in n.ops:
                if not isinstance(op, allowed_ops):
                    raise ValueError("Недопустимый оператор сравнения.")
        def visit_BinOp(self, n):
            if not isinstance(n.op, (ast.BitAnd, ast.BitOr)):
                raise ValueError("Разрешены только & и | для логических операций.")
        def visit_UnaryOp(self, n):
            if not isinstance(n.op, allowed_unary):
                raise ValueError("Недопустимая унарная операция.")
    Checker().visit(node)
    return True

def apply_filter(df: pd.DataFrame, filt: str) -> pd.DataFrame:
    if not filt or not filt.strip():
        return df
    safe_eval_expr(filt)
    try:
        return df.query(filt)
    except Exception as e:
        raise ValueError(f"Ошибка применения фильтра: {e}")

PLOT_SYNONYMS = {
    "hist": ["гистограмма","гист","распределение","распред","histogram","hist"],
    "scatter": ["рассеяние","точечная диаграмма","диаграмма рассеяния","scatter","скаттер"],
    "line": ["линейный","линейный график","временной ряд","line","time series","линия"],
    "bar": ["столбчатый","столбчатая диаграмма","барчарт","bar","категориальная","категорный"],
}

def match_plot_type(text_lower: str):
    for plot, keys in PLOT_SYNONYMS.items():
        for k in keys:
            if k in text_lower:
                return plot
    if "распределени" in text_lower: return "hist"
    if "точечн" in text_lower: return "scatter"
    if "временн" in text_lower: return "line"
    if "столбц" in text_lower or "категор" in text_lower: return "bar"
    return None

def parse_prompt_ru(text: str) -> dict:
    if not text:
        text = ""
    t = text.strip()
    t_low = t.lower()

    plot_type = match_plot_type(t_low)

    def get_val(patterns, src):
        for p in patterns:
            m = re.search(p, src, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    name_re = r"[a-zA-Zа-яА-Я0-9_]+"
    x = get_val([rf"x\s*=\s*({name_re})"], t)
    y = get_val([rf"y\s*=\s*({name_re})"], t)
    col = get_val([rf"col\s*=\s*({name_re})", rf"по столбцу\s+({name_re})"], t)
    bins = get_val([r"bins\s*=\s*([0-9]+)", r"бин[сз]?\s*=\s*([0-9]+)", r"бинов\s*=\s*([0-9]+)"], t)

    filt = None
    m_where = re.search(r"(?:where|где)\s+(.+)$", t, flags=re.IGNORECASE)
    if m_where:
        filt = m_where.group(1).strip()

    m_auto = re.search(r"распределени[ея]\s+([a-zа-я0-9_]+)", t, flags=re.IGNORECASE)
    if m_auto and not (col or x or y):
        col = m_auto.group(1).strip()

    return {
        "plot": plot_type,
        "x": x,
        "y": y,
        "col": col,
        "bins": int(bins) if bins else None,
        "filter": filt
    }

def render_plot(df: pd.DataFrame, plan: dict):
    if plan["filter"]:
        df = apply_filter(df, plan["filter"])
    plot = plan["plot"]
    if plot is None:
        st.warning("Не распознан тип графика. Скажи: гистограмма / рассеяние / линейный / столбчатый.")
        return

    fig = plt.figure(figsize=(7, 4.5))

    if plot == "hist":
        col = plan["col"] or plan["x"] or plan["y"]
        if not col:
            st.error("Для гистограммы укажи столбец: 'по столбцу age' или 'col=age'."); return
        if col not in df.columns:
            st.error(f"Столбец '{col}' не найден в данных."); return
        df[col].dropna().plot(kind="hist", bins=plan["bins"] or 20)
        plt.xlabel(col); plt.ylabel("Частота"); plt.title(f"Гистограмма: {col}")
        st.pyplot(fig)

    elif plot == "scatter":
        if not plan["x"] or not plan["y"]:
            st.error("Для рассеяния укажи x=… и y=…"); return
        if plan["x"] not in df.columns or plan["y"] not in df.columns:
            st.error("Столбцы для x/y не найдены."); return
        plt.scatter(df[plan["x"]], df[plan["y"]], alpha=0.7)
        plt.xlabel(plan["x"]); plt.ylabel(plan["y"]); plt.title(f"Диаграмма рассеяния: {plan['x']} vs {plan['y']}")
        st.pyplot(fig)

    elif plot == "line":
        if not plan["x"] or not plan["y"]:
            st.error("Для линейного графика укажи x=… и y=…"); return
        if plan["x"] not in df.columns or plan["y"] not in df.columns:
            st.error("Столбцы для x/y не найдены."); return
        plt.plot(df[plan["x"]], df[plan["y"]])
        plt.xlabel(plan["x"]); plt.ylabel(plan["y"]); plt.title(f"Линейный график: {plan['y']} по {plan['x']}")
        st.pyplot(fig)

    elif plot == "bar":
        if plan["x"] and plan["y"]:
            x, y = plan["x"], plan["y"]
            if x not in df.columns or y not in df.columns:
                st.error("Столбцы для x/y не найдены."); return
            if pd.api.types.is_numeric_dtype(df[y]):
                agg = df.groupby(x)[y].mean().sort_values(ascending=False)
                agg.plot(kind="bar")
                plt.xlabel(x); plt.ylabel(f"mean({y})"); plt.title(f"Столбчатый: среднее {y} по {x}")
                st.pyplot(fig)
            else:
                cnt = df.groupby([x, y]).size().unstack(fill_value=0).sum(axis=1).sort_values(ascending=False)
                cnt.plot(kind="bar")
                plt.xlabel(x); plt.ylabel("Частота"); plt.title(f"Столбчатый: частоты по {x}")
                st.pyplot(fig)
        elif plan["x"]:
            x = plan["x"]
            if x not in df.columns:
                st.error("Столбец для x не найден."); return
            cnt = df[x].value_counts()
            cnt.plot(kind="bar")
            plt.xlabel(x); plt.ylabel("Частота"); plt.title(f"Столбчатый: распределение {x}")
            st.pyplot(fig)
        else:
            st.error("Для столбчатого укажи хотя бы x=…")
    else:
        st.error(f"Тип графика не поддержан: {plot}")

def generate_python_code(plan: dict) -> str:
    lines = []
    lines.append("import pandas as pd")
    lines.append("import matplotlib.pyplot as plt")
    lines.append("")
    if plan.get("filter"):
        lines.append(f"df_plot = df.query({plan['filter']!r})")
    else:
        lines.append("df_plot = df.copy()")
    lines.append("plt.figure(figsize=(7, 4.5))")
    p = plan.get("plot")

    if p == "hist":
        col = plan.get("col") or plan.get("x") or plan.get("y") or "ВАШ_СТОЛБЕЦ"
        bins = plan.get("bins") or 20
        lines += [
            f"df_plot[{col!r}].dropna().plot(kind='hist', bins={bins})",
            f"plt.xlabel({col!r})",
            "plt.ylabel('Частота')",
            f"plt.title(f'Гистограмма: {col}')",
        ]
    elif p == "scatter":
        x = plan.get("x") or "X"
        y = plan.get("y") or "Y"
        lines += [
            f"plt.scatter(df_plot[{x!r}], df_plot[{y!r}], alpha=0.7)",
            f"plt.xlabel({x!r}); plt.ylabel({y!r})",
            f"plt.title(f'Диаграмма рассеяния: {x} vs {y}')",
        ]
    elif p == "line":
        x = plan.get("x") or "X"
        y = plan.get("y") or "Y"
        lines += [
            f"plt.plot(df_plot[{x!r}], df_plot[{y!r}])",
            f"plt.xlabel({x!r}); plt.ylabel({y!r})",
            f"plt.title(f'Линейный график: {y} по {x}')",
        ]
    elif p == "bar":
        x = plan.get("x")
        y = plan.get("y")
        if x and y:
            lines += [
                f"x, y = {x!r}, {y!r}",
                "if pd.api.types.is_numeric_dtype(df_plot[y]):",
                "    agg = df_plot.groupby(x)[y].mean().sort_values(ascending=False)",
                "    agg.plot(kind='bar')",
                "    plt.xlabel(x); plt.ylabel(f'mean({y})')",
                "    plt.title(f'Столбчатый: среднее {y} по {x}')",
                "else:",
                "    cnt = df_plot.groupby([x, y]).size().unstack(fill_value=0).sum(axis=1).sort_values(ascending=False)",
                "    cnt.plot(kind='bar')",
                "    plt.xlabel(x); plt.ylabel('Частота')",
                "    plt.title(f'Столбчатый: частоты по {x}')",
            ]
        elif x:
            lines += [
                f"x = {x!r}",
                "cnt = df_plot[x].value_counts()",
                "cnt.plot(kind='bar')",
                "plt.xlabel(x); plt.ylabel('Частота')",
                "plt.title(f'Столбчатый: распределение {x}')",
            ]
        else:
            lines.append("# Для столбчатого графика укажите хотя бы x=...")
    else:
        lines.append("# Тип графика не распознан.")
    lines.append("plt.show()")
    return "\n".join(lines)

# ===================== UI: Sidebar =====================

st.sidebar.title("🔧 Режимы")
mode = st.sidebar.radio("Выберите режим", [
    "Визуализация (NL→Plot)",
    "Статистика — WIP",
    "Регрессия — WIP",
    "Корреляция — WIP",
    "Описание данных — WIP"
])
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Загрузить CSV (опционально)", type=["csv"])
use_demo = st.sidebar.toggle("Использовать демо-данные", value=(uploaded is None))

# ===================== Data =====================

if use_demo:
    df = pd.DataFrame({
        "age": np.random.normal(22, 3, 400).round(0).astype(int),
        "study_hours": np.clip(np.random.normal(12, 5, 400), 0, None).round(1),
        "score": np.clip(np.random.normal(74, 11, 400), 0, 100).round(1),
        "height": np.random.normal(170, 8, 400).round(1),
        "weight": np.random.normal(65, 7, 400).round(1),
        "city": np.random.choice(["Tashkent","Samarkand","Bukhara"], 400),
        "subject": np.random.choice(["Math","Stats","CS"], 400),
    })
    df["passed"] = (df["score"] >= 60).astype(int)
else:
    df = load_csv(uploaded)
    if df is None:
        st.stop()

st.caption(f"Данные: {df.shape[0]} строк, {df.shape[1]} столбцов")
st.dataframe(df.head(10), use_container_width=True)

# ===================== Main: only first mode active =====================

if mode != "Визуализация (NL→Plot)":
    st.header(mode)
    st.info("Этот раздел в разработке (Work in Progress).")
    st.stop()

st.header("💬 Естественный язык → график")
st.markdown(
    """
**Типы:** гистограмма, рассеяние (точечная диаграмма), линейный (временной ряд), столбчатый.  
**Параметры:** `x=`, `y=`, `col=`, `бинс=10` (или `bins=10`).  
**Фильтр:** `где ...` (или `where ...`) — выражение формата `pandas.query`, напр.: `где city == 'Tashkent' & age >= 20`.
"""
)

# ---- Input row: two columns; OUTPUTS will be rendered FULL-WIDTH below ----
col_left, col_right = st.columns([1,1], vertical_alignment="top")

with col_left:
    st.markdown("**Примеры запросов:**")
    st.code(
        "построй гистограмму по столбцу score, бинс=30\n"
        "начерти точечную диаграмму: x=height y=weight где city == 'Samarkand'\n"
        "покажи линейный x=age y=score где subject == 'Math'\n"
        "визуализируй столбчатый x=city\n"
        "нарисуй распределение score бинов=40 где subject == 'CS'\n"
        "сделай график: столбчатый x=subject y=score где score >= 80\n"
        "выведи диаграмму: рассеяние x=height y=weight",
        language="text"
    )

with col_right:
    default_prompt = "построй гистограмму по столбцу age, бинс=25 где subject == 'Math'"
    prompt = st.text_area("Опиши нужный график:",
                          value=default_prompt, height=120)

    # button stores result in session, outputs are below columns (full-width)
    if st.button("Скомпилировать запрос и построить график"):
        try:
            plan = parse_prompt_ru(prompt)
            code_text = generate_python_code(plan)
            st.session_state["nl_plot_result"] = {"plan": plan, "code": code_text}
        except Exception as e:
            st.session_state["nl_plot_result"] = {"error": str(e)}

# ---- FULL-WIDTH OUTPUT SECTION ----
st.markdown("---")
res = st.session_state.get("nl_plot_result")
if res:
    if "error" in res:
        st.error(res["error"])
    else:
        st.subheader("План (JSON)")
        st.code(json.dumps(res["plan"], ensure_ascii=False, indent=2), language="json")

        st.subheader("Сгенерированный код (как если бы писали вручную)")
        st.code(res["code"], language="python")

        st.subheader("График")
        try:
            render_plot(df, res["plan"])
        except Exception as e:
            st.error(str(e))
else:
    st.info("Введи запрос и нажми кнопку, чтобы увидеть план, код и график.")
