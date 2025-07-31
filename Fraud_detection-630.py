    elif st.session_state.current_page == "模型信息":
        st.markdown('<div class="main-title">模型信息</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # 完全使用Streamlit原生组件，不使用HTML
        st.subheader(f"🤖 当前模型: {st.session_state.selected_model}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**基本信息:**")
            st.write("• 模型类型: GBDT Pipeline")
            st.write("• 预处理: RobustScaler + GradientBoostingRegressor")
            st.write(f"• 预测结果: {st.session_state.prediction_result:.4f} wt%")
            st.write("• 特征数量: 9个输入特征")
            st.write("• 模型状态: 🟢 正常运行")
        
        with col2:
            st.write("**支持的预测目标:**")
            st.write("• 🔥 **Char Yield:** 焦炭产率预测")
            st.write("• 🛢️ **Oil Yield:** 生物油产率预测")
            st.write("• 💨 **Gas Yield:** 气体产率预测")
        
        st.subheader("📊 特征列表")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        with feature_col1:
            st.write("**Proximate Analysis:**")
            st.write("• M(wt%) - 水分含量")
            st.write("• Ash(wt%) - 灰分含量")
            st.write("• VM(wt%) - 挥发分含量")
        
        with feature_col2:
            st.write("**Ultimate Analysis:**")
            st.write("• O/C - 氧碳原子比")
            st.write("• H/C - 氢碳原子比")
            st.write("• N/C - 氮碳原子比")
        
        with feature_col3:
            st.write("**Pyrolysis Conditions:**")
            st.write("• FT(°C) - 热解温度")
            st.write("• HR(°C/min) - 升温速率")
            st.write("• FR(mL/min) - 载气流量")
        
        st.subheader("📈 当前输入特征值")
        
        # 显示当前特征值
        feature_display_col1, feature_display_col2, feature_display_col3 = st.columns(3)
        features_list = list(st.session_state.feature_values.items())
        
        with feature_display_col1:
            for i in range(0, len(features_list), 3):
                feature, value = features_list[i]
                st.write(f"• **{feature}:** {value:.3f}")
        
        with feature_display_col2:
            for i in range(1, len(features_list), 3):
                if i < len(features_list):
                    feature, value = features_list[i]
                    st.write(f"• **{feature}:** {value:.3f}")
        
        with feature_display_col3:
            for i in range(2, len(features_list), 3):
                if i < len(features_list):
                    feature, value = features_list[i]
                    st.write(f"• **{feature}:** {value:.3f}")
    
    elif st.session_state.current_page == "技术说明":
        st.markdown('<div class="main-title">技术说明</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.subheader("🔬 算法原理")
        st.write("本系统基于**梯度提升决策树(GBDT)**算法构建，采用Pipeline架构集成数据预处理和模型预测。")
        
        st.subheader("🏗️ 系统架构")
        st.write("• **数据预处理:** RobustScaler标准化，对异常值具有较强的鲁棒性")
        st.write("• **机器学习模型:** GradientBoostingRegressor，通过集成多个弱学习器提高预测精度")
        st.write("• **Pipeline集成:** 自动化的数据流处理，确保预测的一致性和可靠性")
        
        st.subheader("📈 模型特点")
        col1, col2 = st.columns(2)
        with col1:
            st.write("• **高精度:** 基于大量实验数据训练，预测精度高")
            st.write("• **鲁棒性:** 对输入数据的噪声和异常值具有较强的容忍性")
        with col2:
            st.write("• **可解释性:** 决策树模型具有良好的可解释性")
            st.write("• **实时性:** 快速响应，支持实时预测")
        
        st.subheader("🎯 应用场景")
        st.write("适用于生物质热解工艺优化、产物产率预测、工艺参数调优等场景。")
        
        st.subheader("⚠️ 使用限制")
        st.warning("• 输入参数应在训练数据范围内，超出范围可能影响预测精度")
        st.warning("• 模型基于特定的实验条件训练，实际应用时需要考虑工艺差异")
        st.warning("• 预测结果仅供参考，实际生产中需要结合实验验证")
    
    elif st.session_state.current_page == "使用指南":
        st.markdown('<div class="main-title">使用指南</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.subheader("📋 操作步骤")
        st.write("1. **选择预测目标:** 点击Char Yield、Oil Yield或Gas Yield按钮选择要预测的产物")
        st.write("2. **输入特征参数:** 在三个特征组中输入相应的数值")
        st.write("3. **执行预测:** 点击"运行预测"按钮获得预测结果")
        st.write("4. **查看结果:** 在右侧面板查看详细的预测信息")
        
        st.subheader("📊 特征参数说明")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.write("#### 🟢 Proximate Analysis")
            st.write("• **M(wt%):** 水分含量，范围 2.75-11.63%")
            st.write("• **Ash(wt%):** 灰分含量，范围 0.41-11.60%")
            st.write("• **VM(wt%):** 挥发分含量，范围 65.70-89.50%")
        
        with param_col2:
            st.write("#### 🟣 Ultimate Analysis")
            st.write("• **O/C:** 氧碳原子比，范围 0.301-0.988")
            st.write("• **H/C:** 氢碳原子比，范围 1.212-1.895")
            st.write("• **N/C:** 氮碳原子比，范围 0.003-0.129")
        
        with param_col3:
            st.write("#### 🟠 Pyrolysis Conditions")
            st.write("• **FT(°C):** 热解温度，范围 300-900°C")
            st.write("• **HR(°C/min):** 升温速率，范围 5-100°C/min")
            st.write("• **FR(mL/min):** 载气流量，范围 0-600 mL/min")
        
        st.subheader("💡 使用技巧")
        tip_col1, tip_col2 = st.columns(2)
        with tip_col1:
            st.info("• **数据质量:** 确保输入数据的准确性，避免明显的错误值")
            st.info("• **参数范围:** 尽量使输入参数在推荐范围内，系统会给出超范围警告")
        with tip_col2:
            st.info("• **结果验证:** 预测结果应结合实际经验进行合理性判断")
            st.info("• **批量预测:** 可以通过修改参数进行多次预测，比较不同条件下的结果")
        
        st.subheader("🔧 功能按钮")
        st.write("• **运行预测:** 基于当前输入参数执行预测")
        st.write("• **重置数据:** 将所有输入参数恢复为默认值")
        st.write("• **执行日志:** 查看系统运行日志和操作记录")
        st.write("• **模型信息:** 查看当前模型的详细信息")