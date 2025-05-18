import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# 데이터 불러오기
df = pd.read_csv('./onlineshopping.csv')

# 'M'으로 시작하는 월별 데이터 컬럼 추출
value_columns = [col for col in df.columns if col.startswith('M') and len(col) >= 7]

# 긴 형식으로 데이터 변환 (melt)
df_long = pd.melt(
    df,
    id_vars=['A 상품군별(1)', 'A 상품군별(2)', 'B 판매매체별(1)'],
    value_vars=value_columns,
    var_name='연월',
    value_name='거래액(백만원)'
)

# 거래액 숫자형 변환
df_long['거래액(백만원)'] = pd.to_numeric(df_long['거래액(백만원)'], errors='coerce')

# 영문명 매핑 (판매매체)
media_map = {
    '10 모바일쇼핑': 'Mobile',
    '20 인터넷쇼핑': 'Internet'
}

# 영문명 매핑 (상품군)
category_map = {
    '000 합계': 'Total',
    '001 컴퓨터 및 주변기기': 'Computer',
    '002 가전·전자·통신기기': 'Electronics',
    '003 서적': 'Books',
    '004 사무·문구': 'Office Supplies',
    '005 의복': 'Clothing',
    '006 신발': 'Shoes',
    '007 가방': 'Bags',
    '008 패션용품 및 액세서리': 'Fashion Accessories',
    '009 스포츠·레저용품': 'Sports & Leisure',
    '010 화장품': 'Cosmetics',
    '011 아동·유아용품': 'Kids & Baby',
    '012 음·식료품': 'Food & Beverages',
    '013 농축수산물': 'Agricultural Products',
    '014 생활용품': 'Household Goods',
    '015 자동차 및 자동차용품': 'Automobiles',
    '016 가구': 'Furniture',
    '017 애완용품': 'Pet Supplies',
    '018 여행 및 교통서비스': 'Travel & Transport',
    '019 문화 및 레저서비스': 'Culture & Leisure',
    '020 이쿠폰서비스': 'E-Coupon Service',
    '021 음식서비스': 'Food Service',
    '022 기타서비스': 'Other Services',
    '023 기타': 'Others'
}

# 매핑 적용
df_long['A 상품군별(1)'] = df_long['A 상품군별(1)'].map(category_map)
df_long['B 판매매체별(1)'] = df_long['B 판매매체별(1)'].map(media_map)

# 컬럼명 변경
df_long.rename(columns={
    'A 상품군별(1)': '상품군코드',
    'A 상품군별(2)': '상품군세부',
    'B 판매매체별(1)': '판매매체',
    '연월': '년월',
    '거래액(백만원)': '거래액_백만원'
}, inplace=True)

# 년월 데이터 전처리
df_long['년월'] = df_long['년월'].str.extract(r'(\d{4}\.\d{2})')
df_long['년월'] = pd.to_datetime(df_long['년월'], format='%Y.%m')

# '판매매체' 열의 결측값을 'Total'로 채우기
df_long.fillna({'판매매체': 'Total'}, inplace=True)

# '소계' 값만 필터링하고 '상품군세부' 열 제거
df_long = df_long[df_long['상품군세부'] == '소계']
df_long.drop(columns='상품군세부', inplace=True)

# 모바일과 인터넷 데이터 병합
df_mobile = df_long[df_long['판매매체'] == 'Mobile'].copy()
df_internet = df_long[df_long['판매매체'] == 'Internet'].copy()
df_merged = pd.merge(
    df_mobile,
    df_internet,
    on=['상품군코드', '년월'],
    suffixes=('_mobile', '_internet')
)

# 컬럼명 변경
df_merged.rename(columns={
    '거래액_백만원_mobile': 'Mobile',
    '거래액_백만원_internet': 'Internet'
}, inplace=True)

# 분석 대상 품목만 선택 (Automobiles, E-Coupon Service, Computer)
categories = ['Automobiles', 'E-Coupon Service', 'Computer']
df_selected = df_merged[df_merged['상품군코드'].isin(categories)].copy()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 자동차 품목 데이터 필터링
df_automobile = df_selected[df_selected['상품군코드'] == 'Automobiles'].copy()

# OLS 회귀모델 생성: Mobile ~ Internet (모바일을 종속변수, 인터넷을 독립변수로)
model_automobile = smf.ols(formula='Mobile ~ Internet', data=df_automobile).fit()
print("[자동차 상품군 회귀모델]")
print(model_automobile.summary())

# 잔차 정규성 검정 (Shapiro-Wilk)
resid_automobile = model_automobile.resid
stat, p_value = shapiro(resid_automobile)
print(f"\nShapiro-Wilk 정규성 검정 p-value: {p_value:.4f}")
print(f"정규성 만족 여부: {'만족' if p_value > 0.05 else '불만족'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test_automobile = het_breuschpagan(resid_automobile, model_automobile.model.exog)
print(f"Breusch-Pagan 등분산성 검정 p-value: {bp_test_automobile[1]:.4f}")
print(f"등분산성 만족 여부: {'만족' if bp_test_automobile[1] > 0.05 else '불만족'}")

# 회귀모델 시각화 (Plotly 버전)
# 산점도와 회귀선 그리기
x_range = np.linspace(df_automobile['Internet'].min(), df_automobile['Internet'].max(), 100)
y_pred = model_automobile.params.iloc[0] + model_automobile.params.iloc[1] * x_range

fig = go.Figure()

# 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_automobile['Internet'], 
        y=df_automobile['Mobile'],
        mode='markers',
        name='자동차 데이터',
        marker=dict(color='blue', size=10, opacity=0.7)
    )
)

# 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='회귀선',
        line=dict(color='red', width=2)
    )
)

# 회귀식과 통계값 표시
r_squared = model_automobile.rsquared
intercept = model_automobile.params.iloc[0]
slope = model_automobile.params.iloc[1]
p_value_slope = model_automobile.pvalues.iloc[1]

regression_eq = f"y = {intercept:.2f} + {slope:.2f}x<br>R² = {r_squared:.3f}<br>p-value = {p_value_slope:.4f}"
test_results = f"정규성 검정(Shapiro-Wilk): p = {p_value:.4f}<br>등분산성 검정(Breusch-Pagan): p = {bp_test_automobile[1]:.4f}"

# 주석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=regression_eq,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

fig.add_annotation(
    x=0.1,
    y=0.85,
    xref="paper",
    yref="paper",
    text=test_results,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='자동차 상품군의 인터넷쇼핑 vs 모바일쇼핑 거래액 회귀분석',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',  # 배경색 설정
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

# 잔차 분석 그래프 (Plotly 버전)
fitted_values = model_automobile.fittedvalues
residuals = model_automobile.resid

fig = make_subplots(rows=1, cols=2, subplot_titles=('잔차 vs 예측값', '잔차 Q-Q 플롯'))

# 첫 번째 서브플롯: 잔차 vs 예측값
fig.add_trace(
    go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='blue', opacity=0.7),
        showlegend=False
    ),
    row=1, col=1
)

# 기준선(y=0) 추가
fig.add_shape(
    type="line",
    x0=min(fitted_values),
    y0=0,
    x1=max(fitted_values),
    y1=0,
    line=dict(color="red", width=2, dash="solid"),
    row=1, col=1
)

# 두 번째 서브플롯: Q-Q 플롯
# 정규성 확률점 계산
from scipy import stats
(osm, osr), _ = stats.probplot(residuals)

fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(color='blue', opacity=0.7),
        name='데이터 포인트',
        showlegend=False
    ),
    row=1, col=2
)

# Q-Q 플롯 참조선 추가
slope, intercept, _, _, _ = stats.linregress(osm, osr)
line_x = np.array([min(osm), max(osm)])
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='참조선',
        showlegend=False
    ),
    row=1, col=2
)

# 그래프 레이아웃 설정
fig.update_layout(
    title_text="잔차 분석 그래프",
    height=500,
    width=900,
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    showlegend=False
)

fig.update_xaxes(title_text="예측값", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="잔차", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_xaxes(title_text="이론적 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="표본 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')

fig.show()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 자동차 품목에 대한 로그 변환
# 데이터에 0이나 음수값이 있는지 확인
min_mobile = df_automobile['Mobile'].min()
min_internet = df_automobile['Internet'].min()
print("Mobile 컬럼의 최소값:", min_mobile)
print("Internet 컬럼의 최소값:", min_internet)

# 로그 변환 변수 생성
df_automobile['log_Mobile'] = np.log(df_automobile['Mobile'])
df_automobile['log_Internet'] = np.log(df_automobile['Internet'])

# 로그-로그 회귀모델 생성
model_automobile_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_automobile).fit()
print("[자동차 상품군 회귀모델 (로그 변환)]")
print(model_automobile_log.summary())

# 잔차 정규성 검정 (Shapiro-Wilk)
resid_automobile_log = model_automobile_log.resid
stat, p_value_log = shapiro(resid_automobile_log)
print(f"\nShapiro-Wilk 정규성 검정 p-value: {p_value_log:.4f}")
print(f"정규성 만족 여부: {'만족' if p_value_log > 0.05 else '불만족'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test_automobile_log = het_breuschpagan(resid_automobile_log, model_automobile_log.model.exog)
print(f"Breusch-Pagan 등분산성 검정 p-value: {bp_test_automobile_log[1]:.4f}")
print(f"등분산성 만족 여부: {'만족' if bp_test_automobile_log[1] > 0.05 else '불만족'}")

# 자동차 상품군 로그-로그 회귀모델 시각화 (Plotly 버전)
x_range = np.linspace(df_automobile['log_Internet'].min(), df_automobile['log_Internet'].max(), 100)
y_pred = model_automobile_log.params.iloc[0] + model_automobile_log.params.iloc[1] * x_range

fig = go.Figure()

# 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_automobile['log_Internet'], 
        y=df_automobile['log_Mobile'],
        mode='markers',
        name='자동차 데이터 (로그 변환)',
        marker=dict(color='blue', size=10, opacity=0.7)
    )
)

# 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='로그-로그 회귀선',
        line=dict(color='red', width=2)
    )
)

# 회귀식과 통계값 표시
r_squared = model_automobile_log.rsquared
intercept = model_automobile_log.params.iloc[0]
slope = model_automobile_log.params.iloc[1]
p_value_slope = model_automobile_log.pvalues.iloc[1]

regression_eq = f"log(y) = {intercept:.2f} + {slope:.2f}log(x)<br>R² = {r_squared:.3f}<br>p-value = {p_value_slope:.4f}"
test_results = f"정규성 검정(Shapiro-Wilk): p = {p_value_log:.4f}<br>등분산성 검정(Breusch-Pagan): p = {bp_test_automobile_log[1]:.4f}"

# 주석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=regression_eq,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

fig.add_annotation(
    x=0.1,
    y=0.85,
    xref="paper",
    yref="paper",
    text=test_results,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='자동차 상품군의 인터넷쇼핑 vs 모바일쇼핑 거래액 회귀분석 (로그-로그 변환)',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (로그 변환)',
    yaxis_title='모바일쇼핑 거래액 (로그 변환)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

# 잔차 분석 그래프 - 로그-로그 모델 (Plotly 버전)
fitted_values = model_automobile_log.fittedvalues
residuals = model_automobile_log.resid

fig = make_subplots(rows=1, cols=2, subplot_titles=('잔차 vs 예측값 (로그-로그 모델)', '잔차 Q-Q 플롯 (로그-로그 모델)'))

# 첫 번째 서브플롯: 잔차 vs 예측값
fig.add_trace(
    go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='blue', opacity=0.7),
        showlegend=False
    ),
    row=1, col=1
)

# 기준선(y=0) 추가
fig.add_shape(
    type="line",
    x0=min(fitted_values),
    y0=0,
    x1=max(fitted_values),
    y1=0,
    line=dict(color="red", width=2, dash="solid"),
    row=1, col=1
)

# 두 번째 서브플롯: Q-Q 플롯
# 정규성 확률점 계산
(osm, osr), _ = stats.probplot(residuals)

fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(color='blue', opacity=0.7),
        name='데이터 포인트',
        showlegend=False
    ),
    row=1, col=2
)

# Q-Q 플롯 참조선 추가
slope, intercept, _, _, _ = stats.linregress(osm, osr)
line_x = np.array([min(osm), max(osm)])
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='참조선',
        showlegend=False
    ),
    row=1, col=2
)

# 그래프 레이아웃 설정
fig.update_layout(
    title_text="잔차 분석 그래프 (로그-로그 모델)",
    height=500,
    width=900,
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    showlegend=False
)

fig.update_xaxes(title_text="예측값 (로그 변환)", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="잔차", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_xaxes(title_text="이론적 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="표본 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')

fig.show()

# 원본 척도로 변환한 그래프 (Plotly 버전)
# 로그-로그 모델을 원본 척도로 변환한 예측선
x_original = np.linspace(df_automobile['Internet'].min(), df_automobile['Internet'].max(), 100)
log_x = np.log(x_original)
log_y_pred = model_automobile_log.params.iloc[0] + model_automobile_log.params.iloc[1] * log_x
y_pred_original = np.exp(log_y_pred)  # 지수 변환으로 원래 스케일로 복원

fig = go.Figure()

# 원본 데이터 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_automobile['Internet'], 
        y=df_automobile['Mobile'],
        mode='markers',
        name='원본 데이터',
        marker=dict(color='blue', size=10, opacity=0.7)
    )
)

# 변환된 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_original,
        y=y_pred_original,
        mode='lines',
        name='로그-로그 모델 (원본 척도)',
        line=dict(color='red', width=2)
    )
)

# 탄력성 해석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=f"탄력성(elasticity): {slope:.2f}<br>(인터넷쇼핑 1% 증가 시 모바일쇼핑 {slope:.2f}% 증가)",
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='자동차 상품군 회귀모델 (로그-로그 모델을 원본 척도로 변환)',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 컴퓨터 품목 데이터 필터링
df_computer = df_selected[df_selected['상품군코드'] == 'Computer'].copy()

# OLS 회귀모델 생성: Mobile ~ Internet
model_computer = smf.ols(formula='Mobile ~ Internet', data=df_computer).fit()
print("[컴퓨터 상품군 회귀모델]")
print(model_computer.summary())

# 잔차 정규성 검정 (Shapiro-Wilk)
resid_computer = model_computer.resid
stat, p_value = shapiro(resid_computer)
print(f"\nShapiro-Wilk 정규성 검정 p-value: {p_value:.4f}")
print(f"정규성 만족 여부: {'만족' if p_value > 0.05 else '불만족'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test_computer = het_breuschpagan(resid_computer, model_computer.model.exog)
print(f"Breusch-Pagan 등분산성 검정 p-value: {bp_test_computer[1]:.4f}")
print(f"등분산성 만족 여부: {'만족' if bp_test_computer[1] > 0.05 else '불만족'}")

# 컴퓨터 상품군 회귀모델 시각화 (Plotly 버전)
x_range = np.linspace(df_computer['Internet'].min(), df_computer['Internet'].max(), 100)
y_pred = model_computer.params.iloc[0] + model_computer.params.iloc[1] * x_range

fig = go.Figure()

# 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_computer['Internet'], 
        y=df_computer['Mobile'],
        mode='markers',
        name='컴퓨터 데이터',
        marker=dict(color='orange', size=10, opacity=0.7)
    )
)

# 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='회귀선',
        line=dict(color='red', width=2)
    )
)

# 회귀식과 통계값 표시
r_squared = model_computer.rsquared
intercept = model_computer.params.iloc[0]
slope = model_computer.params.iloc[1]
p_value_slope = model_computer.pvalues.iloc[1]

regression_eq = f"y = {intercept:.2f} + {slope:.2f}x<br>R² = {r_squared:.3f}<br>p-value = {p_value_slope:.4f}"
test_results = f"정규성 검정(Shapiro-Wilk): p = {p_value:.4f}<br>등분산성 검정(Breusch-Pagan): p = {bp_test_computer[1]:.4f}"

# 주석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=regression_eq,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

fig.add_annotation(
    x=0.1,
    y=0.85,
    xref="paper",
    yref="paper",
    text=test_results,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='컴퓨터 상품군의 인터넷쇼핑 vs 모바일쇼핑 거래액 회귀분석',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

# 잔차 분석 그래프 (Plotly 버전)
fitted_values = model_computer.fittedvalues
residuals = model_computer.resid

fig = make_subplots(rows=1, cols=2, subplot_titles=('잔차 vs 예측값', '잔차 Q-Q 플롯'))

# 첫 번째 서브플롯: 잔차 vs 예측값
fig.add_trace(
    go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='orange', opacity=0.7),
        showlegend=False
    ),
    row=1, col=1
)

# 기준선(y=0) 추가
fig.add_shape(
    type="line",
    x0=min(fitted_values),
    y0=0,
    x1=max(fitted_values),
    y1=0,
    line=dict(color="red", width=2, dash="solid"),
    row=1, col=1
)

# 두 번째 서브플롯: Q-Q 플롯
# 정규성 확률점 계산
(osm, osr), _ = stats.probplot(residuals)

fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(color='orange', opacity=0.7),
        name='데이터 포인트',
        showlegend=False
    ),
    row=1, col=2
)

# Q-Q 플롯 참조선 추가
slope, intercept, _, _, _ = stats.linregress(osm, osr)
line_x = np.array([min(osm), max(osm)])
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='참조선',
        showlegend=False
    ),
    row=1, col=2
)

# 그래프 레이아웃 설정
fig.update_layout(
    title_text="잔차 분석 그래프",
    height=500,
    width=900,
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    showlegend=False
)

fig.update_xaxes(title_text="예측값", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="잔차", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_xaxes(title_text="이론적 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="표본 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')

fig.show()


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 컴퓨터 품목에 대한 로그 변환
# 데이터에 0이나 음수값이 있는지 확인
min_mobile = df_computer['Mobile'].min()
min_internet = df_computer['Internet'].min()
print("Mobile 컬럼의 최소값:", min_mobile)
print("Internet 컬럼의 최소값:", min_internet)

# 로그 변환 변수 생성
df_computer['log_Mobile'] = np.log(df_computer['Mobile'])
df_computer['log_Internet'] = np.log(df_computer['Internet'])

# 로그-로그 회귀모델 생성
model_computer_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_computer).fit()
print("[컴퓨터 상품군 회귀모델 (로그 변환)]")
print(model_computer_log.summary())

# 잔차 정규성 검정 (Shapiro-Wilk)
resid_computer_log = model_computer_log.resid
stat, p_value_log = shapiro(resid_computer_log)
print(f"\nShapiro-Wilk 정규성 검정 p-value: {p_value_log:.4f}")
print(f"정규성 만족 여부: {'만족' if p_value_log > 0.05 else '불만족'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test_computer_log = het_breuschpagan(resid_computer_log, model_computer_log.model.exog)
print(f"Breusch-Pagan 등분산성 검정 p-value: {bp_test_computer_log[1]:.4f}")
print(f"등분산성 만족 여부: {'만족' if bp_test_computer_log[1] > 0.05 else '불만족'}")

# 컴퓨터 상품군 로그-로그 회귀모델 시각화 (Plotly 버전)
x_range = np.linspace(df_computer['log_Internet'].min(), df_computer['log_Internet'].max(), 100)
y_pred = model_computer_log.params.iloc[0] + model_computer_log.params.iloc[1] * x_range

fig = go.Figure()

# 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_computer['log_Internet'], 
        y=df_computer['log_Mobile'],
        mode='markers',
        name='컴퓨터 데이터 (로그 변환)',
        marker=dict(color='orange', size=10, opacity=0.7)
    )
)

# 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='로그-로그 회귀선',
        line=dict(color='red', width=2)
    )
)

# 회귀식과 통계값 표시
r_squared = model_computer_log.rsquared
intercept = model_computer_log.params.iloc[0]
slope = model_computer_log.params.iloc[1]
p_value_slope = model_computer_log.pvalues.iloc[1]

regression_eq = f"log(y) = {intercept:.2f} + {slope:.2f}log(x)<br>R² = {r_squared:.3f}<br>p-value = {p_value_slope:.4f}"
test_results = f"정규성 검정(Shapiro-Wilk): p = {p_value_log:.4f}<br>등분산성 검정(Breusch-Pagan): p = {bp_test_computer_log[1]:.4f}"

# 주석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=regression_eq,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

fig.add_annotation(
    x=0.1,
    y=0.85,
    xref="paper",
    yref="paper",
    text=test_results,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='컴퓨터 상품군의 인터넷쇼핑 vs 모바일쇼핑 거래액 회귀분석 (로그-로그 변환)',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (로그 변환)',
    yaxis_title='모바일쇼핑 거래액 (로그 변환)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

# 잔차 분석 그래프 - 로그-로그 모델 (Plotly 버전)
fitted_values = model_computer_log.fittedvalues
residuals = model_computer_log.resid

fig = make_subplots(rows=1, cols=2, subplot_titles=('잔차 vs 예측값 (로그-로그 모델)', '잔차 Q-Q 플롯 (로그-로그 모델)'))

# 첫 번째 서브플롯: 잔차 vs 예측값
fig.add_trace(
    go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='orange', opacity=0.7),
        showlegend=False
    ),
    row=1, col=1
)

# 기준선(y=0) 추가
fig.add_shape(
    type="line",
    x0=min(fitted_values),
    y0=0,
    x1=max(fitted_values),
    y1=0,
    line=dict(color="red", width=2, dash="solid"),
    row=1, col=1
)

# 두 번째 서브플롯: Q-Q 플롯
# 정규성 확률점 계산
(osm, osr), _ = stats.probplot(residuals)

fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(color='orange', opacity=0.7),
        name='데이터 포인트',
        showlegend=False
    ),
    row=1, col=2
)

# Q-Q 플롯 참조선 추가
slope, intercept, _, _, _ = stats.linregress(osm, osr)
line_x = np.array([min(osm), max(osm)])
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='참조선',
        showlegend=False
    ),
    row=1, col=2
)

# 그래프 레이아웃 설정
fig.update_layout(
    title_text="잔차 분석 그래프 (로그-로그 모델)",
    height=500,
    width=900,
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    showlegend=False
)

fig.update_xaxes(title_text="예측값 (로그 변환)", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="잔차", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_xaxes(title_text="이론적 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="표본 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')

fig.show()

# 원본 척도로 변환한 그래프 (Plotly 버전)
# 로그-로그 모델을 원본 척도로 변환한 예측선
x_original = np.linspace(df_computer['Internet'].min(), df_computer['Internet'].max(), 100)
log_x = np.log(x_original)
log_y_pred = model_computer_log.params.iloc[0] + model_computer_log.params.iloc[1] * log_x
y_pred_original = np.exp(log_y_pred)  # 지수 변환으로 원래 스케일로 복원

fig = go.Figure()

# 원본 데이터 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_computer['Internet'], 
        y=df_computer['Mobile'],
        mode='markers',
        name='원본 데이터',
        marker=dict(color='orange', size=10, opacity=0.7)
    )
)

# 변환된 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_original,
        y=y_pred_original,
        mode='lines',
        name='로그-로그 모델 (원본 척도)',
        line=dict(color='red', width=2)
    )
)

# 탄력성 해석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=f"탄력성(elasticity): {slope:.2f}<br>(인터넷쇼핑 1% 증가 시 모바일쇼핑 {slope:.2f}% 증가)",
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='컴퓨터 상품군 회귀모델 (로그-로그 모델을 원본 척도로 변환)',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 이쿠폰서비스 품목 데이터 필터링
df_ecoupon = df_selected[df_selected['상품군코드'] == 'E-Coupon Service'].copy()

# OLS 회귀모델 생성: Mobile ~ Internet
model_ecoupon = smf.ols(formula='Mobile ~ Internet', data=df_ecoupon).fit()
print("[이쿠폰서비스 상품군 회귀모델]")
print(model_ecoupon.summary())

# 잔차 정규성 검정 (Shapiro-Wilk)
resid_ecoupon = model_ecoupon.resid
stat, p_value = shapiro(resid_ecoupon)
print(f"\nShapiro-Wilk 정규성 검정 p-value: {p_value:.4f}")
print(f"정규성 만족 여부: {'만족' if p_value > 0.05 else '불만족'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test_ecoupon = het_breuschpagan(resid_ecoupon, model_ecoupon.model.exog)
print(f"Breusch-Pagan 등분산성 검정 p-value: {bp_test_ecoupon[1]:.4f}")
print(f"등분산성 만족 여부: {'만족' if bp_test_ecoupon[1] > 0.05 else '불만족'}")

# 이쿠폰서비스 상품군 회귀모델 시각화 (Plotly 버전)
x_range = np.linspace(df_ecoupon['Internet'].min(), df_ecoupon['Internet'].max(), 100)
y_pred = model_ecoupon.params.iloc[0] + model_ecoupon.params.iloc[1] * x_range

fig = go.Figure()

# 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_ecoupon['Internet'], 
        y=df_ecoupon['Mobile'],
        mode='markers',
        name='이쿠폰서비스 데이터',
        marker=dict(color='green', size=10, opacity=0.7)
    )
)

# 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='회귀선',
        line=dict(color='red', width=2)
    )
)

# 회귀식과 통계값 표시
r_squared = model_ecoupon.rsquared
intercept = model_ecoupon.params.iloc[0]
slope = model_ecoupon.params.iloc[1]
p_value_slope = model_ecoupon.pvalues.iloc[1]

regression_eq = f"y = {intercept:.2f} + {slope:.2f}x<br>R² = {r_squared:.3f}<br>p-value = {p_value_slope:.4f}"
test_results = f"정규성 검정(Shapiro-Wilk): p = {p_value:.4f}<br>등분산성 검정(Breusch-Pagan): p = {bp_test_ecoupon[1]:.4f}"

# 주석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=regression_eq,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

fig.add_annotation(
    x=0.1,
    y=0.85,
    xref="paper",
    yref="paper",
    text=test_results,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='이쿠폰서비스 상품군의 인터넷쇼핑 vs 모바일쇼핑 거래액 회귀분석',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

# 잔차 분석 그래프 (Plotly 버전)
fitted_values = model_ecoupon.fittedvalues
residuals = model_ecoupon.resid

fig = make_subplots(rows=1, cols=2, subplot_titles=('잔차 vs 예측값', '잔차 Q-Q 플롯'))

# 첫 번째 서브플롯: 잔차 vs 예측값
fig.add_trace(
    go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='green', opacity=0.7),
        showlegend=False
    ),
    row=1, col=1
)

# 기준선(y=0) 추가
fig.add_shape(
    type="line",
    x0=min(fitted_values),
    y0=0,
    x1=max(fitted_values),
    y1=0,
    line=dict(color="red", width=2, dash="solid"),
    row=1, col=1
)

# 두 번째 서브플롯: Q-Q 플롯
# 정규성 확률점 계산
(osm, osr), _ = stats.probplot(residuals)

fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(color='green', opacity=0.7),
        name='데이터 포인트',
        showlegend=False
    ),
    row=1, col=2
)

# Q-Q 플롯 참조선 추가
slope, intercept, _, _, _ = stats.linregress(osm, osr)
line_x = np.array([min(osm), max(osm)])
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='참조선',
        showlegend=False
    ),
    row=1, col=2
)

# 그래프 레이아웃 설정
fig.update_layout(
    title_text="잔차 분석 그래프",
    height=500,
    width=900,
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    showlegend=False
)

fig.update_xaxes(title_text="예측값", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="잔차", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_xaxes(title_text="이론적 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="표본 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')

fig.show()


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 이쿠폰서비스 품목에 대한 로그 변환
# 데이터에 0이나 음수값이 있는지 확인
min_mobile = df_ecoupon['Mobile'].min()
min_internet = df_ecoupon['Internet'].min()
print("Mobile 컬럼의 최소값:", min_mobile)
print("Internet 컬럼의 최소값:", min_internet)

# 로그 변환 변수 생성
df_ecoupon['log_Mobile'] = np.log(df_ecoupon['Mobile'])
df_ecoupon['log_Internet'] = np.log(df_ecoupon['Internet'])

# 로그-로그 회귀모델 생성
model_ecoupon_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_ecoupon).fit()
print("[이쿠폰서비스 상품군 회귀모델 (로그 변환)]")
print(model_ecoupon_log.summary())

# 잔차 정규성 검정 (Shapiro-Wilk)
resid_ecoupon_log = model_ecoupon_log.resid
stat, p_value_log = shapiro(resid_ecoupon_log)
print(f"\nShapiro-Wilk 정규성 검정 p-value: {p_value_log:.4f}")
print(f"정규성 만족 여부: {'만족' if p_value_log > 0.05 else '불만족'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test_ecoupon_log = het_breuschpagan(resid_ecoupon_log, model_ecoupon_log.model.exog)
print(f"Breusch-Pagan 등분산성 검정 p-value: {bp_test_ecoupon_log[1]:.4f}")
print(f"등분산성 만족 여부: {'만족' if bp_test_ecoupon_log[1] > 0.05 else '불만족'}")

# 이쿠폰서비스 상품군 로그-로그 회귀모델 시각화 (Plotly 버전)
x_range = np.linspace(df_ecoupon['log_Internet'].min(), df_ecoupon['log_Internet'].max(), 100)
y_pred = model_ecoupon_log.params.iloc[0] + model_ecoupon_log.params.iloc[1] * x_range

fig = go.Figure()

# 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_ecoupon['log_Internet'], 
        y=df_ecoupon['log_Mobile'],
        mode='markers',
        name='이쿠폰서비스 데이터 (로그 변환)',
        marker=dict(color='green', size=10, opacity=0.7)
    )
)

# 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='로그-로그 회귀선',
        line=dict(color='red', width=2)
    )
)

# 회귀식과 통계값 표시
r_squared = model_ecoupon_log.rsquared
intercept = model_ecoupon_log.params.iloc[0]
slope = model_ecoupon_log.params.iloc[1]
p_value_slope = model_ecoupon_log.pvalues.iloc[1]

regression_eq = f"log(y) = {intercept:.2f} + {slope:.2f}log(x)<br>R² = {r_squared:.3f}<br>p-value = {p_value_slope:.4f}"
test_results = f"정규성 검정(Shapiro-Wilk): p = {p_value_log:.4f}<br>등분산성 검정(Breusch-Pagan): p = {bp_test_ecoupon_log[1]:.4f}"

# 주석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=regression_eq,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

fig.add_annotation(
    x=0.1,
    y=0.85,
    xref="paper",
    yref="paper",
    text=test_results,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='이쿠폰서비스 상품군의 인터넷쇼핑 vs 모바일쇼핑 거래액 회귀분석 (로그-로그 변환)',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (로그 변환)',
    yaxis_title='모바일쇼핑 거래액 (로그 변환)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

# 잔차 분석 그래프 - 로그-로그 모델 (Plotly 버전)
fitted_values = model_ecoupon_log.fittedvalues
residuals = model_ecoupon_log.resid

fig = make_subplots(rows=1, cols=2, subplot_titles=('잔차 vs 예측값 (로그-로그 모델)', '잔차 Q-Q 플롯 (로그-로그 모델)'))

# 첫 번째 서브플롯: 잔차 vs 예측값
fig.add_trace(
    go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='green', opacity=0.7),
        showlegend=False
    ),
    row=1, col=1
)

# 기준선(y=0) 추가
fig.add_shape(
    type="line",
    x0=min(fitted_values),
    y0=0,
    x1=max(fitted_values),
    y1=0,
    line=dict(color="red", width=2, dash="solid"),
    row=1, col=1
)

# 두 번째 서브플롯: Q-Q 플롯
# 정규성 확률점 계산
(osm, osr), _ = stats.probplot(residuals)

fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(color='green', opacity=0.7),
        name='데이터 포인트',
        showlegend=False
    ),
    row=1, col=2
)

# Q-Q 플롯 참조선 추가
slope, intercept, _, _, _ = stats.linregress(osm, osr)
line_x = np.array([min(osm), max(osm)])
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='참조선',
        showlegend=False
    ),
    row=1, col=2
)

# 그래프 레이아웃 설정
fig.update_layout(
    title_text="잔차 분석 그래프 (로그-로그 모델)",
    height=500,
    width=900,
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    showlegend=False
)

fig.update_xaxes(title_text="예측값 (로그 변환)", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="잔차", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_xaxes(title_text="이론적 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')
fig.update_yaxes(title_text="표본 분위수", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.6)')

fig.show()

# 원본 척도로 변환한 그래프 (Plotly 버전)
# 로그-로그 모델을 원본 척도로 변환한 예측선
x_original = np.linspace(df_ecoupon['Internet'].min(), df_ecoupon['Internet'].max(), 100)
log_x = np.log(x_original)
log_y_pred = model_ecoupon_log.params.iloc[0] + model_ecoupon_log.params.iloc[1] * log_x
y_pred_original = np.exp(log_y_pred)  # 지수 변환으로 원래 스케일로 복원

fig = go.Figure()

# 원본 데이터 산점도 추가
fig.add_trace(
    go.Scatter(
        x=df_ecoupon['Internet'], 
        y=df_ecoupon['Mobile'],
        mode='markers',
        name='원본 데이터',
        marker=dict(color='green', size=10, opacity=0.7)
    )
)

# 변환된 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_original,
        y=y_pred_original,
        mode='lines',
        name='로그-로그 모델 (원본 척도)',
        line=dict(color='red', width=2)
    )
)

# 탄력성 해석 추가
fig.add_annotation(
    x=0.1,
    y=0.95,
    xref="paper",
    yref="paper",
    text=f"탄력성(elasticity): {slope:.2f}<br>(인터넷쇼핑 1% 증가 시 모바일쇼핑 {slope:.2f}% 증가)",
    showarrow=False,
    font=dict(size=12),
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    borderpad=4,
    align="left"
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='이쿠폰서비스 상품군 회귀모델 (로그-로그 모델을 원본 척도로 변환)',
    title_font_size=15,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    )
)

fig.show()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 세 가지 상품군 회귀분석 결과를 한꺼번에 시각화 (원본 데이터) (Plotly 버전)
# 각 상품군별 데이터 준비
categories = ['Automobiles', 'E-Coupon Service', 'Computer']
colors = ['blue', 'green', 'orange']
markers = ['circle', 'square', 'diamond']
models = []
dataframes = []

fig = go.Figure()

# 각 상품군별 데이터셋과 모델 준비
for i, category in enumerate(categories):
    # 데이터 필터링
    df_temp = df_selected[df_selected['상품군코드'] == category].copy()
    
    # 표준 OLS 모델 생성
    model = smf.ols(formula='Mobile ~ Internet', data=df_temp).fit()
    
    # 리스트에 저장
    dataframes.append(df_temp)
    models.append(model)
    
    # 산점도 그리기
    fig.add_trace(
        go.Scatter(
            x=df_temp['Internet'],
            y=df_temp['Mobile'],
            mode='markers',
            name=f'{category} 데이터',
            marker=dict(
                color=colors[i],
                symbol=markers[i],
                size=10,
                opacity=0.7
            )
        )
    )
    
    # 회귀선 그리기
    x_range = np.linspace(df_temp['Internet'].min(), df_temp['Internet'].max(), 100)
    y_pred = model.params.iloc[0] + model.params.iloc[1] * x_range
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name=f'{category} 회귀선',
            line=dict(color=colors[i], width=2)
        )
    )
    
    # 회귀식 표시
    r_squared = model.rsquared
    y_pos = 0.95 - (i * 0.1)  # 각 회귀식의 위치를 조절
    regression_eq = f"{category}: y = {model.params.iloc[0]:.2f} + {model.params.iloc[1]:.2f}x (R² = {r_squared:.3f})"
    
    fig.add_annotation(
        x=0.05,
        y=y_pos,
        xref="paper",
        yref="paper",
        text=regression_eq,
        showarrow=False,
        font=dict(size=12, color=colors[i]),
        bgcolor="white",
        bordercolor=colors[i],
        borderwidth=1,
        borderpad=4,
        align="left"
    )

# 그래프 레이아웃 설정
fig.update_layout(
    title='상품군별 인터넷쇼핑 vs 모바일쇼핑 거래액 회귀분석 비교 (원본 데이터)',
    title_font_size=16,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    legend=dict(
        x=0.7,
        y=0.95,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.8)'
    ),
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    height=700,
    width=1000
)

fig.show()

# 회귀분석 결과 요약 출력
for i, category in enumerate(categories):
    print(f"\n[{category} 회귀분석 결과 요약 (원본 데이터)]")
    print(f"회귀식: y = {models[i].params.iloc[0]:.4f} + {models[i].params.iloc[1]:.4f}x")
    print(f"결정계수(R²): {models[i].rsquared:.4f}")
    print(f"p-value: {models[i].pvalues.iloc[1]:.6f}")


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 세 가지 상품군 로그-로그 회귀분석 결과를 원본 척도로 변환하여 한꺼번에 시각화 (Plotly 버전)
# 각 상품군별 데이터 준비
categories = ['Automobiles', 'E-Coupon Service', 'Computer']
colors = ['blue', 'green', 'orange']
markers = ['circle', 'square', 'diamond']
models_log = []
dataframes = []

fig = go.Figure()

# 각 상품군별 데이터셋과 로그-로그 모델 준비
for i, category in enumerate(categories):
    # 데이터 필터링
    df_temp = df_selected[df_selected['상품군코드'] == category].copy()
    
    # 로그 변환 변수 생성
    df_temp['log_Mobile'] = np.log(df_temp['Mobile'])
    df_temp['log_Internet'] = np.log(df_temp['Internet'])
    
    # 로그-로그 회귀모델 생성
    model_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_temp).fit()
    
    # 리스트에 저장
    dataframes.append(df_temp)
    models_log.append(model_log)
    
    # 원본 데이터를 이용한 산점도 그리기
    fig.add_trace(
        go.Scatter(
            x=df_temp['Internet'],
            y=df_temp['Mobile'],
            mode='markers',
            name=f'{category} 데이터',
            marker=dict(
                color=colors[i],
                symbol=markers[i],
                size=10,
                opacity=0.7
            )
        )
    )
    
    # 로그-로그 회귀선을 원본 스케일로 변환하여 그리기
    x_original = np.linspace(df_temp['Internet'].min(), df_temp['Internet'].max(), 100)
    log_x = np.log(x_original)
    log_y_pred = model_log.params.iloc[0] + model_log.params.iloc[1] * log_x
    y_pred_original = np.exp(log_y_pred)  # 지수 변환으로 원래 스케일로 복원
    
    fig.add_trace(
        go.Scatter(
            x=x_original,
            y=y_pred_original,
            mode='lines',
            name=f'{category} 로그-로그 회귀선',
            line=dict(color=colors[i], width=2)
        )
    )
    
    # 회귀식 표시 (로그-로그 모델)
    r_squared = model_log.rsquared
    slope = model_log.params.iloc[1]
    intercept = model_log.params.iloc[0]
    y_pos = 0.95 - (i * 0.1)  # 각 회귀식의 위치를 조절
    regression_eq = f"{category}: y = exp({intercept:.2f}) * x^({slope:.2f}) (R² = {r_squared:.3f})"
    
    fig.add_annotation(
        x=0.05,
        y=y_pos,
        xref="paper",
        yref="paper",
        text=regression_eq,
        showarrow=False,
        font=dict(size=12, color=colors[i]),
        bgcolor="white",
        bordercolor=colors[i],
        borderwidth=1,
        borderpad=4,
        align="left"
    )

# 그래프 레이아웃 설정
fig.update_layout(
    title='상품군별 인터넷쇼핑 vs 모바일쇼핑 거래액 회귀분석 비교 (원본 데이터, 로그-로그 변환 모델)',
    title_font_size=16,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    legend=dict(
        x=0.7,
        y=0.95,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.8)'
    ),
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    height=700,
    width=1000
)

fig.show()

# 회귀분석 결과 요약 출력
for i, category in enumerate(categories):
    print(f"\n[{category} 회귀분석 결과 요약 (로그-로그 변환)]")
    print(f"회귀식: log(Mobile) = {models_log[i].params.iloc[0]:.4f} + {models_log[i].params.iloc[1]:.4f} * log(Internet)")
    print(f"해석: 인터넷쇼핑 거래액이 1% 증가할 때, 모바일쇼핑 거래액은 약 {models_log[i].params.iloc[1]:.4f}% 증가")
    print(f"결정계수(R²): {models_log[i].rsquared:.4f}")
    print(f"p-value: {models_log[i].pvalues.iloc[1]:.6f}")


# 로그 변환 데이터의 스피어만 상관계수 분석
# 자동차 품목
df_automobile = df_selected[df_selected['상품군코드'] == 'Automobiles'].copy()
df_automobile['log_Mobile'] = np.log(df_automobile['Mobile'])
df_automobile['log_Internet'] = np.log(df_automobile['Internet'])
spearman_corr_auto, spearman_p_auto = stats.spearmanr(df_automobile['log_Internet'], df_automobile['log_Mobile'])

# 이쿠폰서비스 품목
df_ecoupon = df_selected[df_selected['상품군코드'] == 'E-Coupon Service'].copy()
df_ecoupon['log_Mobile'] = np.log(df_ecoupon['Mobile'])
df_ecoupon['log_Internet'] = np.log(df_ecoupon['Internet'])
spearman_corr_ecoupon, spearman_p_ecoupon = stats.spearmanr(df_ecoupon['log_Internet'], df_ecoupon['log_Mobile'])

# 컴퓨터 품목
df_computer = df_selected[df_selected['상품군코드'] == 'Computer'].copy()
df_computer['log_Mobile'] = np.log(df_computer['Mobile'])
df_computer['log_Internet'] = np.log(df_computer['Internet'])
spearman_corr_computer, spearman_p_computer = stats.spearmanr(df_computer['log_Internet'], df_computer['log_Mobile'])

# 결과 출력
categories = ['Automobiles', 'E-Coupon Service', 'Computer']
spearman_corrs = [spearman_corr_auto, spearman_corr_ecoupon, spearman_corr_computer]
spearman_p_values = [spearman_p_auto, spearman_p_ecoupon, spearman_p_computer]

print("\n[로그 변환 데이터의 스피어만 상관계수 분석 결과]")
for i, category in enumerate(categories):
    print(f"\n{category}:")
    print(f"Spearman 상관계수: {spearman_corrs[i]:.4f} (p={spearman_p_values[i]:.4f})")
    sig = "유의함" if spearman_p_values[i] < 0.05 else "유의하지 않음"
    print(f"통계적 유의성 (α=0.05): {sig}")


# 회귀계수 계산 (로그-로그 모델)
model_automobile_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_automobile).fit()
model_ecoupon_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_ecoupon).fit()
model_computer_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_computer).fit()

log_regression_slopes = [model_automobile_log.params.iloc[1], model_ecoupon_log.params.iloc[1], model_computer_log.params.iloc[1]]

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 회귀계수 계산 (로그-로그 모델)
model_automobile_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_automobile).fit()
model_ecoupon_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_ecoupon).fit()
model_computer_log = smf.ols(formula='log_Mobile ~ log_Internet', data=df_computer).fit()

# 각 상품군별 계수 준비
categories = ['Automobiles', 'E-Coupon Service', 'Computer']
colors = ['blue', 'green', 'orange']  # 각 상품군별 동일한 색상 사용

# 스피어만 상관계수
spearman_corrs = [spearman_corr_auto, spearman_corr_ecoupon, spearman_corr_computer]
spearman_p_values = [spearman_p_auto, spearman_p_ecoupon, spearman_p_computer]

# 로그-로그 회귀계수
log_regression_slopes = [
    model_automobile_log.params.iloc[1], 
    model_ecoupon_log.params.iloc[1], 
    model_computer_log.params.iloc[1]
]
regression_p_values = [
    model_automobile_log.pvalues.iloc[1], 
    model_ecoupon_log.pvalues.iloc[1], 
    model_computer_log.pvalues.iloc[1]
]

# 두 개의 차트를 나란히 그리기 (Plotly 버전)
fig = make_subplots(
    rows=1, 
    cols=2, 
    subplot_titles=('상품군별 Spearman 상관계수', '상품군별 회귀계수(탄력성)'),
    horizontal_spacing=0.15  # 두 차트 사이 간격 증가
)

# 첫 번째 서브플롯: 스피어만 상관계수
for i, category in enumerate(categories):
    fig.add_trace(
        go.Bar(
            x=[category],
            y=[spearman_corrs[i]],
            name=category,
            marker_color=colors[i],
            text=[f'{spearman_corrs[i]:.3f}'],
            textposition='outside',
            width=0.6,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 통계적 유의성 표시 (p < 0.05)
    if spearman_p_values[i] < 0.05:
        fig.add_annotation(
            x=category,
            y=spearman_corrs[i] - 0.05,
            text="*",
            showarrow=False,
            font=dict(size=20),
            row=1, col=1
        )

# 두 번째 서브플롯: 로그-로그 회귀계수
for i, category in enumerate(categories):
    fig.add_trace(
        go.Bar(
            x=[category],
            y=[log_regression_slopes[i]],
            name=category,
            marker_color=colors[i],
            text=[f'{log_regression_slopes[i]:.3f}'],
            textposition='outside',
            width=0.6
        ),
        row=1, col=2
    )
    
    # 통계적 유의성 표시 (p < 0.05)
    if regression_p_values[i] < 0.05:
        fig.add_annotation(
            x=category,
            y=log_regression_slopes[i] - 0.05,
            text="*",
            showarrow=False,
            font=dict(size=20),
            row=1, col=2
        )

# 그래프 레이아웃 설정
fig.update_layout(
    title='상품군별 상관계수 및 회귀계수(탄력성) 비교',
    title_font_size=16,
    legend_title='상품군',
    barmode='group',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    showlegend=True,
    height=600,  # 높이 증가
    width=1400,  # 너비 더 증가
    margin=dict(l=50, r=50, t=100, b=150),  # 모든 여백 증가, 특히 하단 여백
)

# x축 레이아웃 설정 - 첫 번째 차트
fig.update_xaxes(
    tickangle=0,  # 레이블을 기울이지 않음
    tickfont=dict(size=12),  # 글꼴 크기
    row=1, col=1
)

# x축 레이아웃 설정 - 두 번째 차트
fig.update_xaxes(
    tickangle=0,  # 레이블을 기울이지 않음
    tickfont=dict(size=12),  # 글꼴 크기
    row=1, col=2
)

# y축 설정
fig.update_yaxes(title_text="Spearman 상관계수", row=1, col=1, range=[0, 1.1])
fig.update_yaxes(title_text="로그-로그 회귀계수(탄력성)", row=1, col=2)

# 서브플롯 제목 위치 조정
for i in fig['layout']['annotations'][:2]:  # 첫 두 개의 주석이 서브플롯 제목임
    i['y'] = 1.05  # 제목 위치를 위로 조정

fig.show()











''''''


print("1. 원본 데이터 기반 통합 회귀모델")

# 통합 회귀모델 생성 (원본 데이터)
# statsmodels의 C() 함수를 사용하여 상품군코드를 범주형 변수로 지정
# 상호작용 항을 포함하기 위해 '*' 연산자 사용
unified_model = smf.ols(
    formula='Mobile ~ Internet * C(상품군코드)', 
    data=df_selected
).fit()

print("[통합 회귀모델 요약]")
print(unified_model.summary())

# 기준범주 확인
print("\n기준 범주:", sorted(df_selected['상품군코드'].unique())[0])

# 정규성 검정 (Shapiro-Wilk)
resid_unified = unified_model.resid
stat, p_value = shapiro(resid_unified)
print(f"\nShapiro-Wilk 정규성 검정 p-value: {p_value:.4f}")
print(f"정규성 만족 여부: {'만족' if p_value > 0.05 else '불만족'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test_unified = het_breuschpagan(resid_unified, unified_model.model.exog)
print(f"Breusch-Pagan 등분산성 검정 p-value: {bp_test_unified[1]:.4f}")
print(f"등분산성 만족 여부: {'만족' if bp_test_unified[1] > 0.05 else '불만족'}")

# 잔차 분석 그래프 (Plotly 버전)
fitted_values = unified_model.fittedvalues
residuals = unified_model.resid

fig = make_subplots(rows=1, cols=2, subplot_titles=('잔차 vs 예측값', '잔차 Q-Q 플롯'))

# 품목별로 색상을 구분하기 위한 준비
colors = {'Automobiles': 'blue', 'E-Coupon Service': 'green', 'Computer': 'orange'}

# 각 품목별 잔차 표시
for category in categories:
    # 해당 품목 인덱스
    idx = df_selected[df_selected['상품군코드'] == category].index
    
    # 첫 번째 서브플롯: 잔차 vs 예측값
    fig.add_trace(
        go.Scatter(
            x=fitted_values[idx],
            y=residuals[idx],
            mode='markers',
            marker=dict(color=colors[category], opacity=0.7),
            name=f'{category} 잔차'
        ),
        row=1, col=1
    )

# 기준선(y=0) 추가
fig.add_shape(
    type="line",
    x0=min(fitted_values),
    y0=0,
    x1=max(fitted_values),
    y1=0,
    line=dict(color="red", width=2, dash="solid"),
    row=1, col=1
)

# 두 번째 서브플롯: Q-Q 플롯
(osm, osr), _ = stats.probplot(residuals)

fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(color='purple', opacity=0.7),
        name='Q-Q 플롯',
        showlegend=True
    ),
    row=1, col=2
)

# Q-Q 플롯 참조선 추가
slope, intercept, _, _, _ = stats.linregress(osm, osr)
line_x = np.array([min(osm), max(osm)])
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='참조선'
    ),
    row=1, col=2
)

# 그래프 레이아웃 설정
fig.update_layout(
    title_text="통합 회귀모델 잔차 분석 그래프",
    height=500,
    width=900,
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.update_xaxes(title_text="예측값", row=1, col=1)
fig.update_yaxes(title_text="잔차", row=1, col=1)
fig.update_xaxes(title_text="이론적 분위수", row=1, col=2)
fig.update_yaxes(title_text="표본 분위수", row=1, col=2)

fig.show()

# 품목별 통합 회귀모델 시각화
fig = go.Figure()

# 각 품목별 데이터와 회귀선 추가
for category in categories:
    # 해당 품목 데이터 필터링
    df_cat = df_selected[df_selected['상품군코드'] == category]
    
    # 산점도 추가
    fig.add_trace(
        go.Scatter(
            x=df_cat['Internet'],
            y=df_cat['Mobile'],
            mode='markers',
            name=f'{category} 데이터',
            marker=dict(color=colors[category], size=10, opacity=0.7)
        )
    )
    
    # 해당 품목에 대한 예측선 생성을 위한 X 값 준비
    x_range = np.linspace(df_cat['Internet'].min(), df_cat['Internet'].max(), 100)
    
    # 통합 모델을 이용한 예측값 계산을 위해 가상 데이터프레임 생성
    pred_df = pd.DataFrame({
        'Internet': x_range,
        '상품군코드': category
    })
    
    # predict 메서드를 사용하여 예측값 계산
    y_pred = unified_model.predict(pred_df)
    
    # 회귀선 추가
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name=f'{category} 회귀선',
            line=dict(color=colors[category], width=2)
        )
    )

# 그래프 레이아웃 설정
fig.update_layout(
    title='품목별 인터넷쇼핑 vs 모바일쇼핑 거래액 (통합 회귀모델)',
    title_font_size=16,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    height=600,
    width=1000
)

fig.show()


print("2. 로그-로그 변환 통합 회귀모델")


# 로그 변환 변수 생성
df_selected['log_Mobile'] = np.log(df_selected['Mobile'])
df_selected['log_Internet'] = np.log(df_selected['Internet'])

# 로그-로그 통합 모델
unified_log_model = smf.ols(
    formula='log_Mobile ~ log_Internet * C(상품군코드)', 
    data=df_selected
).fit()

print("\n[통합 로그-로그 회귀모델 요약]")
print(unified_log_model.summary())

# 정규성 검정 (Shapiro-Wilk)
resid_unified_log = unified_log_model.resid
stat, p_value_log = shapiro(resid_unified_log)
print(f"\nShapiro-Wilk 정규성 검정 p-value: {p_value_log:.4f}")
print(f"정규성 만족 여부: {'만족' if p_value_log > 0.05 else '불만족'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test_unified_log = het_breuschpagan(resid_unified_log, unified_log_model.model.exog)
print(f"Breusch-Pagan 등분산성 검정 p-value: {bp_test_unified_log[1]:.4f}")
print(f"등분산성 만족 여부: {'만족' if bp_test_unified_log[1] > 0.05 else '불만족'}")

# 로그-로그 모델 잔차 분석 그래프
fitted_values_log = unified_log_model.fittedvalues
residuals_log = unified_log_model.resid

fig = make_subplots(rows=1, cols=2, subplot_titles=('잔차 vs 예측값 (로그-로그 모델)', '잔차 Q-Q 플롯 (로그-로그 모델)'))

# 각 품목별 잔차 표시
for category in categories:
    # 해당 품목 인덱스
    idx = df_selected[df_selected['상품군코드'] == category].index
    
    # 첫 번째 서브플롯: 잔차 vs 예측값
    fig.add_trace(
        go.Scatter(
            x=fitted_values_log[idx],
            y=residuals_log[idx],
            mode='markers',
            marker=dict(color=colors[category], opacity=0.7),
            name=f'{category} 잔차'
        ),
        row=1, col=1
    )

# 기준선(y=0) 추가
fig.add_shape(
    type="line",
    x0=min(fitted_values_log),
    y0=0,
    x1=max(fitted_values_log),
    y1=0,
    line=dict(color="red", width=2, dash="solid"),
    row=1, col=1
)

# 두 번째 서브플롯: Q-Q 플롯
(osm, osr), _ = stats.probplot(residuals_log)

fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(color='purple', opacity=0.7),
        name='Q-Q 플롯',
        showlegend=True
    ),
    row=1, col=2
)

# Q-Q 플롯 참조선 추가
slope, intercept, _, _, _ = stats.linregress(osm, osr)
line_x = np.array([min(osm), max(osm)])
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='참조선'
    ),
    row=1, col=2
)

# 그래프 레이아웃 설정
fig.update_layout(
    title_text="통합 로그-로그 회귀모델 잔차 분석 그래프",
    height=500,
    width=900,
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.update_xaxes(title_text="예측값 (로그 변환)", row=1, col=1)
fig.update_yaxes(title_text="잔차", row=1, col=1)
fig.update_xaxes(title_text="이론적 분위수", row=1, col=2)
fig.update_yaxes(title_text="표본 분위수", row=1, col=2)

fig.show()

# 로그-로그 모델 시각화
fig = go.Figure()

# 각 품목별 로그 변환된 데이터와 회귀선 추가
for category in categories:
    # 해당 품목 데이터 필터링
    df_cat = df_selected[df_selected['상품군코드'] == category]
    
    # 산점도 추가 (로그 변환 데이터)
    fig.add_trace(
        go.Scatter(
            x=df_cat['log_Internet'],
            y=df_cat['log_Mobile'],
            mode='markers',
            name=f'{category} 데이터 (로그 변환)',
            marker=dict(color=colors[category], size=10, opacity=0.7)
        )
    )
    
    # 해당 품목에 대한 예측선 생성을 위한 X 값 준비
    x_log_range = np.linspace(df_cat['log_Internet'].min(), df_cat['log_Internet'].max(), 100)
    
    # 통합 모델을 이용한 예측값 계산을 위해 가상 데이터프레임 생성
    pred_df = pd.DataFrame({
        'log_Internet': x_log_range,
        '상품군코드': category
    })
    
    # predict 메서드를 사용하여 예측값 계산
    y_log_pred = unified_log_model.predict(pred_df)
    
    # 회귀선 추가
    fig.add_trace(
        go.Scatter(
            x=x_log_range,
            y=y_log_pred,
            mode='lines',
            name=f'{category} 회귀선 (로그-로그)',
            line=dict(color=colors[category], width=2)
        )
    )

# 그래프 레이아웃 설정
fig.update_layout(
    title='품목별 인터넷쇼핑 vs 모바일쇼핑 거래액 (로그-로그 통합 회귀모델)',
    title_font_size=16,
    xaxis_title='인터넷쇼핑 거래액 (로그 변환)',
    yaxis_title='모바일쇼핑 거래액 (로그 변환)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    height=600,
    width=1000
)

fig.show()

# 로그-로그 모델을 원본 척도로 변환하여 시각화
fig = go.Figure()

# 각 품목별 원본 데이터와 변환된 회귀선 추가
for category in categories:
    # 해당 품목 데이터 필터링
    df_cat = df_selected[df_selected['상품군코드'] == category]
    
    # 원본 산점도 추가
    fig.add_trace(
        go.Scatter(
            x=df_cat['Internet'],
            y=df_cat['Mobile'],
            mode='markers',
            name=f'{category} 원본 데이터',
            marker=dict(color=colors[category], size=10, opacity=0.7)
        )
    )
    
    # 해당 품목에 대한 예측을 위한 X 값 준비 (원본 척도)
    x_original = np.linspace(df_cat['Internet'].min(), df_cat['Internet'].max(), 100)
    
    # 로그 변환
    x_log = np.log(x_original)
    
    # 통합 모델을 이용한 예측값 계산을 위해 가상 데이터프레임 생성
    pred_df = pd.DataFrame({
        'log_Internet': x_log,
        '상품군코드': category
    })
    
    # predict 메서드를 사용하여 로그 척도에서의 예측값 계산
    y_log_pred = unified_log_model.predict(pred_df)
    
    # 원본 척도로 변환 (지수 변환)
    y_pred_original = np.exp(y_log_pred)
    
    # 변환된 회귀선 추가
    fig.add_trace(
        go.Scatter(
            x=x_original,
            y=y_pred_original,
            mode='lines',
            name=f'{category} 회귀선 (변환)',
            line=dict(color=colors[category], width=2)
        )
    )

# 그래프 레이아웃 설정
fig.update_layout(
    title='품목별 인터넷쇼핑 vs 모바일쇼핑 거래액 (로그-로그 모델 원본 척도 변환)',
    title_font_size=16,
    xaxis_title='인터넷쇼핑 거래액 (백만원)',
    yaxis_title='모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    height=600,
    width=1000
)

fig.show()


print("3. 품목별 탄력성(회귀계수) 비교")


# 로그-로그 모델에서 각 품목별 탄력성(기울기) 추출
params = unified_log_model.params
base_category = sorted(df_selected['상품군코드'].unique())[0]
print(f"기준 품목: {base_category}")

# 기준 범주의 기울기(탄력성)
base_elasticity = params['log_Internet']
print(f"\n{base_category}의 탄력성: {base_elasticity:.4f}")

# 다른 품목들의 기울기(탄력성)
for category in categories:
    if category != base_category:
        interaction_term = f"log_Internet:C(상품군코드)[T.{category}]"
        if interaction_term in params:
            elasticity = base_elasticity + params[interaction_term]
            print(f"{category}의 탄력성: {elasticity:.4f} (기준품목과의 차이: {params[interaction_term]:.4f})")

# 품목별 탄력성 시각화
fig = go.Figure()

# 기준 품목의 탄력성
elasticities = [base_elasticity]
elasticity_labels = [base_category]

# 다른 품목들의 탄력성 계산
for category in categories:
    if category != base_category:
        interaction_term = f"log_Internet:C(상품군코드)[T.{category}]"
        if interaction_term in params:
            elasticity = base_elasticity + params[interaction_term]
            elasticities.append(elasticity)
            elasticity_labels.append(category)

# 바 차트 그리기
fig.add_trace(go.Bar(
    x=elasticity_labels,
    y=elasticities,
    text=[f"{e:.3f}" for e in elasticities],
    textposition='outside',
    marker_color=[colors[cat] for cat in elasticity_labels],
    width=0.5
))

# 그래프 레이아웃 설정
fig.update_layout(
    title='품목별 탄력성 (인터넷쇼핑 1% 증가 시 모바일쇼핑 증가율)',
    title_font_size=16,
    xaxis_title='상품군',
    yaxis_title='탄력성 (회귀계수)',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)',
        zerolinecolor='black'
    ),
    height=500,
    width=800
)

fig.show()


print("4. 통계적 유의성 검정")


# 계수의 통계적 유의성 확인
print("\n로그-로그 모델 계수의 p-값:")
for name, pvalue in unified_log_model.pvalues.items():
    print(f"{name}: {pvalue:.4f} {'(유의함)' if pvalue < 0.05 else '(유의하지 않음)'}")

# ANOVA 분석 - 품목이 유의한 영향을 미치는지 검정
from statsmodels.stats.anova import anova_lm

# 제약 모델 (품목 구분 없음)
restricted_model = smf.ols(formula='log_Mobile ~ log_Internet', data=df_selected).fit()

# ANOVA 테스트 실행
anova_results = anova_lm(restricted_model, unified_log_model)
print("\nANOVA 분석 결과 (품목 영향 검정):")
print(anova_results)

# 품목별 더미변수들이 함께 유의한지 테스트 (F-test)
from statsmodels.stats.anova import anova_lm

# 제약 모델 (품목 구분 없음)
restricted_model = smf.ols(formula='log_Mobile ~ log_Internet', data=df_selected).fit()

# ANOVA 테스트 실행
anova_results = anova_lm(restricted_model, unified_log_model)
print("\nANOVA 분석 결과 (품목 영향 검정):")
print(anova_results)

# AIC, BIC를 통한 모델 비교
print("\n모델 비교 (AIC, BIC):")
print(f"통합 로그-로그 모델 - AIC: {unified_log_model.aic:.2f}, BIC: {unified_log_model.bic:.2f}")
print(f"제약 모델 (품목 없음) - AIC: {restricted_model.aic:.2f}, BIC: {restricted_model.bic:.2f}")


print("5. 모델 성능 평가")


# R-squared 비교
print(f"통합 로그-로그 모델 - R²: {unified_log_model.rsquared:.4f}, 수정된 R²: {unified_log_model.rsquared_adj:.4f}")
print(f"제약 모델 (품목 없음) - R²: {restricted_model.rsquared:.4f}, 수정된 R²: {restricted_model.rsquared_adj:.4f}")

# 잔차의 평균제곱오차(MSE) 계산
mse_unified = np.mean(unified_log_model.resid**2)
mse_restricted = np.mean(restricted_model.resid**2)

print(f"통합 로그-로그 모델 - MSE: {mse_unified:.6f}")
print(f"제약 모델 (품목 없음) - MSE: {mse_restricted:.6f}")

# 실제값 vs 예측값 시각화
fig = go.Figure()

# 각 품목별 실제값과 예측값 비교
for category in categories:
    df_cat = df_selected[df_selected['상품군코드'] == category]
    
    # 예측값 계산
    predictions = unified_log_model.predict(df_cat)
    
    # 원본 척도로 변환된 예측값
    predictions_original = np.exp(predictions)
    
    fig.add_trace(
        go.Scatter(
            x=df_cat['Mobile'],
            y=predictions_original,
            mode='markers',
            name=f'{category} 예측',
            marker=dict(color=colors[category], size=10, opacity=0.7)
        )
    )

# 대각선 추가 (완벽한 예측 참조선)
max_val = max(df_selected['Mobile'].max(), np.exp(predictions).max())
min_val = min(df_selected['Mobile'].min(), np.exp(predictions).min())

fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='완벽한 예측',
        line=dict(color='black', width=2, dash='dash')
    )
)

# 그래프 레이아웃 설정
fig.update_layout(
    title='실제값 vs 예측값 (통합 로그-로그 모델, 원본 척도)',
    title_font_size=16,
    xaxis_title='실제 모바일쇼핑 거래액 (백만원)',
    yaxis_title='예측 모바일쇼핑 거래액 (백만원)',
    legend_title='범례',
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.6)'
    ),
    height=600,
    width=800
)


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy

# 1. patsy로 디자인 매트릭스 만들기 (회귀식 그대로 사용)
y, X = patsy.dmatrices(
    'log_Mobile ~ log_Internet * C(상품군코드)', 
    data=df_selected, 
    return_type='dataframe'
)

# 2. VIF 계산
vif_data = pd.DataFrame()
vif_data["변수"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 3. 결과 출력
print("\n[다중공선성 (VIF) 확인]")
print(vif_data)
