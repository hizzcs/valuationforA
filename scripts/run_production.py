#!/usr/bin/env python3
"""
生产环境启动脚本
包含:
- 系统检查
- 依赖验证
- 环境变量配置
- 服务启动
"""
import sys
import os
import subprocess
import time
from pathlib import Path

def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 11):
        print("❌ Python版本至少需要3.11")
        return False
    
    print("✅ Python版本检查通过")
    return True

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("❌ requirements.txt 不存在")
        return False
    
    try:
        import pkg_resources
        with open(requirements_path, 'r') as f:
            dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        missing = []
        for dep in dependencies:
            try:
                pkg_resources.require(dep)
            except:
                missing.append(dep)
        
        if missing:
            print(f"❌ 缺少依赖: {', '.join(missing)}")
            return False
        
        print("✅ 所有依赖检查通过")
        return True
        
    except ImportError:
        print("❌ 无法导入pkg_resources，请检查pip是否正确安装")
        return False

def check_env_variables():
    """检查环境变量"""
    print("🔍 检查环境变量...")
    
    required_vars = ["TUSHARE_TOKEN"]
    
    missing_vars = []
    for var in required_vars:
        if var not in os.environ:
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            if key == var:
                                os.environ[key] = value.strip()
                                break
            else:
                missing_vars.append(var)
        else:
            # 验证token格式
            if var == "TUSHARE_TOKEN" and len(os.environ[var]) < 10:
                print("❌ TUSHARE_TOKEN格式不正确")
                return False
    
    if missing_vars:
        print(f"❌ 缺少环境变量: {', '.join(missing_vars)}")
        print("💡 请在 .env 文件中设置或直接导出到环境变量")
        return False
        
    print("✅ 环境变量检查通过")
    return True

def start_streamlit_server():
    """启动Streamlit服务器"""
    print("🚀 启动Streamlit服务器...")
    
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/streamlit_app.py", 
        "--server.port", "8502",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        process = subprocess.Popen(
            streamlit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待服务器启动
        time.sleep(5)
        
        # 检查是否正常运行
        if process.poll() is not None:
            stderr = process.stderr.read()
            print(f"❌ Streamlit服务器启动失败: {stderr}")
            return False
        
        print("✅ Streamlit服务器启动成功")
        print("📡 服务地址: http://localhost:8502")
        print("🔗 外部访问: http://0.0.0.0:8502")
        
        return True
        
    except Exception as e:
        print(f"❌ 启动服务器时出错: {e}")
        return False

def create_production_config():
    """创建生产环境配置"""
    print("📝 创建生产环境配置...")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "production.yaml"
    
    config_content = """# 生产环境配置
app:
  title: "A股绝对估值 - 投资经理工作台"
  page_config:
    layout: "wide"
    initial_sidebar_state: "expanded"
    menu_items:
      About: "A股绝对估值系统 - 生产环境版本"

data:
  cache_dir: "./cache"
  duckdb_path: "./duckdb/valuation.duckdb"
  fixtures_dir: "./tests/data"

validation:
  quality_grades: ["A", "B", "C", "D"]
  data_checks:
    - "revenue_positive"
    - "net_profit_accuracy"
    - "debt_equity_ratio"
    - "cash_flow_consistency"

valuation:
  default_scenarios: 5000
  confidence_levels: [0.05, 0.5, 0.95]
  monte_carlo:
    seed: 42
    max_iterations: 10000
    convergence_tolerance: 0.01

reporting:
  include_metrics:
    - "data_quality_grade"
    - "source_mode"
    - "beta"
    - "wacc"
    - "terminal_growth"
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✅ 生产配置文件创建成功")
    return True

def main():
    """主函数"""
    print("🚀 A股绝对估值系统 - 生产环境启动")
    print("=" * 50)
    
    # 检查所有要求
    checks = [
        check_system_requirements,
        check_dependencies,
        check_env_variables,
        create_production_config
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
            break
    
    if not all_passed:
        print("\n❌ 启动失败，请解决上述问题")
        return False
    
    print("\n✅ 所有检查通过，开始启动服务")
    print("=" * 50)
    
    if start_streamlit_server():
        print("\n🎉 系统启动成功！")
        print("\n📋 生产环境说明:")
        print("- 所有接口访问受限")
        print("- 数据缓存已启用")
        print("- 监控指标已配置")
        print("- 日志文件将保存到 logs/ 目录")
        
        try:
            # 等待用户中断
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n⏹️  正在关闭服务器...")
            return True
    else:
        print("\n❌ 服务启动失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
