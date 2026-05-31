# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px

# Import các thư viện xử lý lượng tử của Qiskit
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# =====================================================================
# STAGE 1: CLASSICAL DATA ENGINE
# =====================================================================
def get_financial_metrics(tickers):
    # Tự động thêm hậu tố '.VN' cho các mã chứng khoán Việt Nam nếu chưa có
    formatted_tickers = [t + '.VN' if not t.endswith('.VN') else t for t in tickers]
    
    # Tải dữ liệu giá đóng cửa trong 2 năm gần nhất từ Yahoo Finance
    df = yf.download(formatted_tickers, start="2024-01-01", end="2026-01-01")
    
    if df.empty:
        raise ValueError("Không tải được dữ liệu giá từ Yahoo Finance. Vui lòng kiểm tra lại mã cổ phiếu.")
        
    # Xử lý MultiIndex columns khi tải nhiều mã
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.levels[0]:
            data = df['Close']
        elif 'Adj Close' in df.columns.levels[0]:
            data = df['Adj Close']
        else:
            raise ValueError("Không tìm thấy cột giá đóng cửa trong dữ liệu.")
    else:
        if 'Close' in df.columns:
            data = df['Close']
        elif 'Adj Close' in df.columns:
            data = df['Adj Close']
        else:
            data = df
            
    # Tính tỷ suất sinh lời hàng ngày và loại bỏ giá trị rỗng (NaN)
    returns = data.pct_change().dropna()
    
    # Trích xuất danh sách mã cổ phiếu đã sắp xếp (bỏ hậu tố '.VN')
    sorted_tickers = [col.replace('.VN', '') for col in data.columns]
    
    # Lấy giá đóng cửa cuối cùng của từng cổ phiếu (dạng VND)
    last_prices = {col.replace('.VN', ''): data[col].iloc[-1] for col in data.columns}
    
    # Tính toán lợi nhuận kỳ vọng và ma trận hiệp phương sai
    mu = returns.mean().to_numpy()
    sigma = returns.cov().to_numpy()
    
    if np.any(np.isnan(mu)) or np.any(np.isnan(sigma)):
        raise ValueError("Dữ liệu chứa giá trị NaN sau khi xử lý lợi nhuận. Vui lòng kiểm tra lại.")
        
    return mu, sigma, sorted_tickers, last_prices

# =====================================================================
# STAGE 2: QUANTUM COMPUTING OPTIMIZER (QAOA CORE)
# =====================================================================
def solve_portfolio_via_qaoa(mu, sigma, risk_factor, budget):
    # 1. Định nghĩa bài toán Tối ưu hóa danh mục với Qiskit Optimization
    portfolio_problem = PortfolioOptimization(
        expected_returns=mu, 
        covariances=sigma, 
        risk_factor=risk_factor, 
        budget=budget
    )
    
    # 2. Tự động chuyển đổi sang dạng Chương trình bậc hai (QUBO)
    quadratic_program = portfolio_problem.to_quadratic_program()
    
    # 3. Cấu hình trình lấy mẫu (Sampler) để giả lập môi trường lượng tử (V2 StatevectorSampler)
    backend_sampler = StatevectorSampler()
    
    # 4. Khởi tạo thuật toán lượng tử biến phân QAOA với bộ tối ưu cổ điển COBYLA
    qaoa_algorithm = QAOA(sampler=backend_sampler, optimizer=COBYLA(), reps=2)
    
    # 5. Thực thi tối ưu toán tử và giải bài toán bằng bộ giải lượng tử
    quantum_optimizer = MinimumEigenOptimizer(qaoa_algorithm)
    execution_result = quantum_optimizer.solve(quadratic_program)
    
    # 6. Giải mã kết quả nhị phân trả về dạng mảng nhãn (ví dụ: [1, 0, 1, 0])
    selected_assets = portfolio_problem.interpret(execution_result)
    
    return selected_assets

# =====================================================================
# STAGE 3: INTERACTIVE USER INTERFACE (STREAMLIT FRONTEND)
# =====================================================================
def main():
    # Cấu hình trang giao diện chính
    st.set_page_config(
        page_title="AI-Quantum Portfolio Optimizer", 
        page_icon="⚡", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ AI-Quantum Portfolio Optimizer using QAOA")
    st.caption("Ứng dụng Điện toán lượng tử giải bài toán tối ưu hóa tài chính Markowitz | Đề tài AI-Quantum Challenge 2026")
    st.markdown("---")
    
    # Giao diện chia làm 2 cột: Cột trái cấu hình đầu vào, Cột phải hiển thị kết quả
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Cấu hình danh mục")
        
        # Chọn rổ cổ phiếu
        selected_tickers = st.multiselect(
            "Chọn rổ cổ phiếu kiểm thử (VN30):", 
            options=['FPT', 'HPG', 'VCB', 'VNM', 'MSN', 'VIC', 'TCB', 'MWG'], 
            default=['FPT', 'HPG', 'VCB', 'VNM']
        )
        
        # Thêm liên kết theo dõi real-time của các cổ phiếu đã chọn
        if selected_tickers:
            with st.expander("📈 Xem giá real-time các cổ phiếu đã chọn", expanded=True):
                for t in selected_tickers:
                    st.markdown(f"- **{t}**: [FireAnt 📊](https://fireant.vn/dashboard/content/symbols/{t}) | [CafeF 📰](https://cafef.vn/co-phieu/{t}.chn)")
        
        # Thiết lập nguồn vốn hiện có
        capital = st.slider(
            "Nguồn vốn đầu tư hiện có (VND):",
            min_value=10_000_000,
            max_value=1_000_000_000,
            value=100_000_000,
            step=10_000_000,
            help="Tổng số vốn VND bạn muốn phân bổ vào danh mục đầu tư này."
        )
        st.markdown(f"💰 **Tổng vốn đầu tư:** `{capital:,.0f} VND`")
        st.markdown("---")
        
        # Thiết lập tham số động
        risk_factor = st.slider(
            "Mức độ ngại rủi ro (Risk Factor - q):",
            min_value=0.01,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Giá trị càng cao thì mô hình càng ưu tiên giảm thiểu rủi ro (biến động) hơn là lợi nhuận."
        )
        
        # Budget tối đa giới hạn bởi số lượng mã cổ phiếu được chọn
        max_budget = max(1, len(selected_tickers))
        budget = st.slider(
            "Số lượng cổ phiếu tối đa trong danh mục (Budget - B):",
            min_value=1,
            max_value=max_budget,
            value=min(2, max_budget),
            step=1,
            help="Số lượng cổ phiếu mà thuật toán lượng tử được phép chọn để phân bổ vốn."
        )
        
        run_optimization = st.button("🚀 Thực thi tối ưu lượng tử", type="primary")
        
    with col2:
        st.subheader("📊 Kết quả phân bổ vốn")
        
        if run_optimization:
            if len(selected_tickers) < 2:
                st.error("Vui lòng chọn tối thiểu 2 mã cổ phiếu để thực hiện tối ưu hóa danh mục!")
                return
                
            with st.spinner("Hệ thống đang cào dữ liệu vĩ mô và ánh xạ toán tử Ising Hamiltonian sang mạch lượng tử QAOA..."):
                try:
                    # BƯỚC 1: Gọi tầng Data xử lý dữ liệu cổ điển
                    mu, sigma, sorted_tickers, last_prices = get_financial_metrics(selected_tickers)
                    
                    # BƯỚC 2: Gọi tầng Quantum Core mô phỏng QAOA
                    quantum_selection = solve_portfolio_via_qaoa(mu, sigma, risk_factor, budget)
                    
                    # BƯỚC 3: Xử lý hiển thị kết quả trực quan ra màn hình
                    st.success("Tối ưu hóa thành công trên Trình mô phỏng Lượng tử StatevectorSampler!")
                    
                    # Lọc ra các mã cổ phiếu được thuật toán lựa chọn (quantum_selection là mảng index các tài sản được chọn, ví dụ: [1, 2])
                    final_portfolio = [sorted_tickers[idx] for idx in quantum_selection]
                    
                    if len(final_portfolio) == 0:
                        st.warning("Mô hình không tìm thấy tổ hợp nào tối ưu với tham số rủi ro hiện tại. Hãy điều chỉnh lại số lượng cổ phiếu chọn.")
                    else:
                        # Tính toán phân bổ vốn
                        allocation_ratio = 100.0 / len(final_portfolio)
                        allocated_capital_per_asset = float(capital) / len(final_portfolio)
                        
                        # Tạo bảng dữ liệu kết quả (dữ liệu số thô để định dạng bằng Streamlit column_config)
                        prices = [last_prices.get(t, 1.0) for t in final_portfolio]
                        quantities = [allocated_capital_per_asset / p for p in prices]
                        
                        # Làm tròn lô 100 cổ phiếu (quy chuẩn giao dịch sàn HOSE)
                        lots_100 = [max(100, round(q / 100) * 100) if q >= 50 else 0 for q in quantities]
                        
                        result_df = pd.DataFrame({
                            'Mã Cổ Phiếu': final_portfolio,
                            'Giá đóng cửa (VND)': prices,
                            'Tỷ lệ phân bổ (%)': [allocation_ratio] * len(final_portfolio),
                            'Số tiền phân bổ (VND)': [allocated_capital_per_asset] * len(final_portfolio),
                            'Số lượng cổ phiếu (CP)': quantities,
                            'Khuyên dùng mua (Lô 100 CP)': lots_100,
                            'Theo dõi giá': [f"https://fireant.vn/dashboard/content/symbols/{t}" for t in final_portfolio]
                        })
                        
                        # Hiển thị bảng dữ liệu định dạng chuyên nghiệp
                        st.write("### 📝 Chi tiết phân bổ danh mục:")
                        st.dataframe(
                            result_df, 
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Mã Cổ Phiếu": st.column_config.TextColumn(help="Mã chứng khoán được lựa chọn bởi QAOA"),
                                "Giá đóng cửa (VND)": st.column_config.NumberColumn(format="%d VND", help="Giá đóng cửa thực tế cuối phiên gần nhất"),
                                "Tỷ lệ phân bổ (%)": st.column_config.NumberColumn(format="%.1f%%"),
                                "Số tiền phân bổ (VND)": st.column_config.NumberColumn(format="%d VND", help="Số tiền phân bổ cụ thể theo tổng nguồn vốn"),
                                "Số lượng cổ phiếu (CP)": st.column_config.NumberColumn(format="%.1f CP"),
                                "Khuyên dùng mua (Lô 100 CP)": st.column_config.NumberColumn(format="%d CP", help="Khuyên dùng làm tròn lô 100 cổ phiếu theo quy tắc sàn HOSE"),
                                "Theo dõi giá": st.column_config.LinkColumn(display_text="Xem biểu đồ 📈", help="Đường dẫn đến đồ thị kỹ thuật và giá real-time trên FireAnt")
                            }
                        )
                        
                        # Vẽ biểu đồ tròn Plotly đẹp mắt và tương tác
                        fig = px.pie(
                            result_df, 
                            values='Tỷ lệ phân bổ (%)', 
                            names='Mã Cổ Phiếu', 
                            title='Tỷ lệ phân bổ vốn tối ưu trong danh mục',
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_traces(textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình tính toán mạch lượng tử: {str(e)}")
        else:
            st.info("Nhấn nút 'Thực thi tối ưu lượng tử' ở cột trái để bắt đầu chạy mô phỏng mạch QAOA.")

if __name__ == "__main__":
    main()