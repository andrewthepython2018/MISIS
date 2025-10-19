import re
import io
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import linregress, shapiro

st.set_page_config(page_title="NL → Plotter & Stats", page_icon="📊", layout="wide")

# ===================== Utilities =====================

def load_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")))

def infer_numeric(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def infer_categorical(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

def safe_eval_expr(expr: str):
    """
    Статическая проверка ограниченного булева выражения для фильтра:
    имена столбцов, числа, строковые литералы, сравнения, &, |, скобки, унарные +/-/not.
    Без вызовов функций и атрибутов.
    """
    allowed_ops = (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE)
    allowed_bools = (ast.And, ast.Or)
    allowed_unary = (ast.USub, ast.UAdd, ast.Not)
    allowed_nodes = (
        ast.Expression, ast.BoolOp, ast.BinOp, ast.Compare, ast.Name, ast.Load,
        ast.Constant, ast.UnaryOp, ast.Subscript, ast.Tuple, ast.List, ast.Dict
    )
    try:
        node = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError(f"Некорректное выражение фильтра: {e}")

    class Checker(ast.NodeVisitor):
        def generic_visit(self, n):
            if not isinstance(n, allowed_nodes):
                raise ValueError(f"Запрещенная конструкция в фильтре: {type(n).__name__}")
            super().generic_visit(n)
        def visit_Attribute(self, n):
            raise ValueError("Атрибуты запрещены в фильтре.")
        def visit_Call(self, n):
            raise ValueError("Вызовы функций запрещены в фильтре.")
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

def parse_prompt(text: str) -> dict:
    """
    Примеры:
      - "построй гистограмму по столбцу age, bins=30"
      - "scatter x=height y=weight где city == 'Tashkent'"
      - "line x=date y=sales"
      - "bar x=city" или "bar x=subject y=score"
    """
    t = (text or "").lower().strip()

    plot_type = None
    if any(k in t for k in ["гист", "histogram", "hist"]):
        plot_type = "hist"
    elif any(k in t for k in ["scatter", "рассеян", "диаграмму рассеяния"]):
        plot_type = "scatter"
    elif any(k in t for k in ["line", "линейн"]):
        plot_type = "line"
    elif any(k in t for k in ["bar", "столбч", "категор"]):
        plot_type = "bar"

    def get_val(patterns):
        for p in patterns:
            m = re.search(p, t)
            if m:
                return m.group(1).strip()
        return None

    x = get_val([r"x\s*=\s*([a-z0-9_]+)"])
    y = get_val([r"y\s*=\s*([a-z0-9_]+)"])
    col = get_val([r"col\s*=\s*([a-z0-9_]+)", r"по столбцу\s+([a-z0-9_]+)"])
    bins = get_val([r"bins\s*=\s*([0-9]+)", r"бин[сз]?\s*=\s*([0-9]+)"])

    filt = None
    m_where = re.search(r"(?:where|где)\s+(.+)$", text or "", flags=re.IGNORECASE)
    if m_where:
        filt = m_where.group(1).strip()

    plan = {
        "plot": plot_type,
        "x": x,
        "y": y,
        "col": col,
        "bins": int(bins) if bins else None,
        "filter": filt
    }
    return plan

def render_plot(df: pd.DataFrame, plan: dict):
    if plan["filter"]:
        df = apply_filter(df, plan["filter"])
    plot = plan["plot"]
    if plot is None:
        st.warning("Не распознан тип графика. Скажи 'гистограмма', 'scatter', 'line' или 'bar'.")
        return

    fig = plt.figure(figsize=(7, 4.5))
    if plot == "hist":
        col = plan["col"] or plan["x"] or plan["y"]
        if not col:
            st.error("Для гистограммы укажи столбец: 'по столбцу age' или 'col=age'.")
            return
        if col not in df.columns:
            st.error(f"Столбец '{col}' не найден.")
            return
        df[col].dropna().plot(kind="hist", bins=plan["bins"] or 20)
        plt.xlabel(col); plt.ylabel("Count"); plt.title(f"Histogram of {col}")
        st.pyplot(fig)

    elif plot == "scatter":
        if not plan["x"] or not plan["y"]:
            st.error("Для scatter укажи x=… и y=…")
            return
        if plan["x"] not in df.columns or plan["y"] not in df.columns:
            st.error("Столбцы для x/y не найдены.")
            return
        plt.scatter(df[plan["x"]], df[plan["y"]], alpha=0.7)
        plt.xlabel(plan["x"]); plt.ylabel(plan["y"]); plt.title(f"Scatter: {plan['x']} vs {plan['y']}")
        st.pyplot(fig)

    elif plot == "line":
        if not plan["x"] or not plan["y"]:
            st.error("Для line укажи x=… и y=…")
            return
        if plan["x"] not in df.columns or plan["y"] not in df.columns:
            st.error("Столбцы для x/y не найдены.")
            return
        plt.plot(df[plan["x"]], df[plan["y"]])
        plt.xlabel(plan["x"]); plt.ylabel(plan["y"]); plt.title(f"Line: {plan['y']} over {plan['x']}")
        st.pyplot(fig)

    elif plot == "bar":
        if plan["x"] and plan["y"]:
            x, y = plan["x"], plan["y"]
            if x not in df.columns or y not in df.columns:
                st.error("Столбцы для x/y не найдены.")
                return
            if pd.api.types.is_numeric_dtype(df[y]):
                agg = df.groupby(x)[y].mean().sort_values(ascending=False)
                agg.plot(kind="bar")
                plt.ylabel(f"mean({y})"); plt.title(f"Bar: mean({y}) by {x}"); plt.xlabel(x)
                st.pyplot(fig)
            else:
                cnt = df.groupby([x, y]).size().unstack(fill_value=0).sum(axis=1).sort_values(ascending=False)
                cnt.plot(kind="bar")
                plt.ylabel("Count"); plt.title(f"Bar: count by {x}"); plt.xlabel(x)
                st.pyplot(fig)
        elif plan["x"]:
            x = plan["x"]
            if x not in df.columns:
                st.error("Столбец для x не найден.")
                return
            cnt = df[x].value_counts()
            cnt.plot(kind="bar")
            plt.xlabel(x); plt.ylabel("Count"); plt.title(f"Bar: counts of {x}")
            st.pyplot(fig)
        else:
            st.error("Для bar укажи x=… (и опционально y=…).")
    else:
        st.error(f"Тип графика не поддержан: {plot}")

def corr_heatmap(ax, corr, labels):
    im = ax.imshow(corr, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
    ax.set_title("Correlation matrix")
    return im

# ===================== Sidebar & data =====================

st.sidebar.title("🔧 Настройки")
mode = st.sidebar.radio("Режим", ["Визуализация (NL→Plot)", "Статистика", "Регрессия", "Корреляция", "Описание данных"])
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Загрузить CSV", type=["csv"])
demo = st.sidebar.toggle("Использовать демо-данные", value=(uploaded is None))

if demo:
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

# ===================== Modes =====================

if mode == "Визуализация (NL→Plot)":
    st.header("💬 Natural Language → Plot")
    prompt = st.text_area(
        "Опиши желаемый график естественным языком",
        value="Построй гистограмму по столбцу age, bins=25 где subject == 'Math'"
    )
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Построить"):
            try:
                plan = parse_prompt(prompt)
                st.markdown("**План (JSON):**")
                st.code(json.dumps(plan, ensure_ascii=False, indent=2), language="json")
                render_plot(df, plan)
            except Exception as e:
                st.error(str(e))
    with col2:
        st.markdown("**Примеры запросов:**")
        st.code(
            "гистограмма по столбцу score, bins=30\n"
            "scatter x=height y=weight где city == 'Samarkand'\n"
            "line x=age y=score где subject == 'Stats'\n"
            "bar x=city\n"
            "bar x=subject y=score где score >= 80",
            language="text"
        )

elif mode == "Статистика":
    st.header("📈 Базовая статистика")
    num_cols = infer_numeric(df)
    cat_cols = infer_categorical(df)

    st.subheader("Сводная таблица (describe)")
    st.dataframe(df[num_cols].describe().T, use_container_width=True)

    st.subheader("Boxplot выбранного числового столбца")
    sel = st.selectbox("Столбец", num_cols)
    fig = plt.figure(figsize=(6,4))
    plt.boxplot(df[sel].dropna(), vert=True, labels=[sel])
    plt.title(f"Boxplot: {sel}")
    st.pyplot(fig)

    st.subheader("Гистограмма")
    bins = st.slider("bins", min_value=5, max_value=60, value=20, step=1)
    fig2 = plt.figure(figsize=(6,4))
    plt.hist(df[sel].dropna(), bins=bins)
    plt.xlabel(sel); plt.ylabel("Count"); plt.title(f"Histogram: {sel}")
    st.pyplot(fig2)

    if cat_cols:
        st.subheader("Частоты категориального столбца")
        cat = st.selectbox("Категориальный столбец", cat_cols)
        vc = df[cat].value_counts()
        fig3 = plt.figure(figsize=(7,4))
        vc.plot(kind="bar")
        plt.title(f"Counts of {cat}"); plt.xlabel(cat); plt.ylabel("Count")
        st.pyplot(fig3)

elif mode == "Регрессия":
    st.header("📐 Линейная регрессия (X → Y)")
    num_cols = infer_numeric(df)
    c1, c2 = st.columns(2)
    with c1:
        x = st.selectbox("Признак X (числовой)", num_cols, index=0 if num_cols else None)
    with c2:
        y = st.selectbox("Целевая Y (числовая)", num_cols, index=1 if len(num_cols)>1 else 0)

    if x and y:
        sub = df[[x, y]].dropna()
        if len(sub) < 3:
            st.warning("Недостаточно данных после очистки.")
        else:
            slope, intercept, r, p, stderr = linregress(sub[x], sub[y])
            st.markdown(f"**Модель:**  \n`{y} = {slope:.4f} * {x} + {intercept:.4f}`")
            st.markdown(f"**R²:** {r**2:.4f} &nbsp;&nbsp; **p-value:** {p:.3e} &nbsp;&nbsp; **stderr (slope):** {stderr:.4f}")

            fig = plt.figure(figsize=(7,4))
            plt.scatter(sub[x], sub[y], alpha=0.6)
            xs = np.linspace(sub[x].min(), sub[x].max(), 200)
            plt.plot(xs, slope*xs + intercept)
            plt.xlabel(x); plt.ylabel(y); plt.title("Линейная регрессия")
            st.pyplot(fig)

            st.subheader("Анализ остатков")
            residuals = sub[y] - (slope*sub[x] + intercept)
            fig2 = plt.figure(figsize=(7,3.8))
            plt.hist(residuals, bins=30)
            plt.title("Распределение остатков"); plt.xlabel("Residual"); plt.ylabel("Count")
            st.pyplot(fig2)

            sample = residuals.sample(min(500, len(residuals)), random_state=0) if len(residuals)>500 else residuals
            W, pval = shapiro(sample)
            st.markdown(f"**Shapiro–Wilk (остатки):** W={W:.3f}, p-value={pval:.3e}")

elif mode == "Корреляция":
    st.header("🔗 Корреляционный анализ")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        st.warning("Нужно минимум два числовых признака.")
    else:
        sel = st.multiselect("Числовые столбцы", num_cols, default=num_cols[:min(6, len(num_cols))])
        if sel:
            sub = df[sel].dropna()
            corr = sub.corr().values
            fig, ax = plt.subplots(figsize=(0.8*len(sel)+2, 0.8*len(sel)+2))
            im = corr_heatmap(ax, corr, sel)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

            st.subheader("Сильные корреляции (|r| ≥ 0.6)")
            pairs = []
            for i in range(len(sel)):
                for j in range(i+1, len(sel)):
                    pairs.append(((sel[i], sel[j]), corr[i,j]))
            strong = sorted([p for p in pairs if abs(p[1])>=0.6], key=lambda z: -abs(z[1]))
            if strong:
                for (a,b), r in strong[:10]:
                    st.write(f"- **{a} — {b}: r={r:.3f}**")
            else:
                st.write("Нет пар с |r| ≥ 0.6")

            st.subheader("Диаграмма рассеяния для пары")
            c1, c2 = st.columns(2)
            with c1:
                a = st.selectbox("X", sel, index=0)
            with c2:
                b = st.selectbox("Y", sel, index=1 if len(sel)>1 else 0)
            show_line = st.toggle("Показать тренд-линию", value=True)
            sub2 = df[[a,b]].dropna()
            fig2 = plt.figure(figsize=(7,4))
            plt.scatter(sub2[a], sub2[b], alpha=0.6)
            if show_line and len(sub2)>2:
                s, itc, *_ = linregress(sub2[a], sub2[b])
                xs = np.linspace(sub2[a].min(), sub2[a].max(), 200)
                plt.plot(xs, s*xs + itc)
            plt.xlabel(a); plt.ylabel(b); plt.title(f"Scatter: {a} vs {b}")
            st.pyplot(fig2)

elif mode == "Описание данных":
    st.header("🔍 Описание и качество данных")
    st.subheader("Типы столбцов")
    dtypes = pd.DataFrame({"dtype": df.dtypes.astype(str)})
    st.dataframe(dtypes)

    st.subheader("Пропуски")
    na = df.isna().sum().sort_values(ascending=False)
    st.dataframe(na.to_frame("NA_count"))

    st.subheader("Память")
    mem = df.memory_usage(deep=True).sum() / (1024**2)
    st.write(f"Оценка памяти: **{mem:.2f} MB**")

    st.subheader("Примеры категориальных распределений")
    cats = [c for c in df.columns if df[c].dtype == object]
    if cats:
        cat = st.selectbox("Категориальный столбец", cats)
        vc = df[cat].value_counts().head(20)
        fig = plt.figure(figsize=(7,4))
        vc.plot(kind="bar")
        plt.title(f"Top-20 {cat}"); plt.xlabel(cat); plt.ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("Категориальных столбцов не обнаружено.")
