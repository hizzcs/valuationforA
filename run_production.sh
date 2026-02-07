#!/bin/bash

# A股绝对估值系统 - 生产环境启动脚本
# 适用于投资经理生产环境的自动化部署和管理

set -e  # 遇到错误立即停止

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 日志配置
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/production_$(date +%Y%m%d_%H%M%S).log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：打印信息
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# 函数：打印警告
warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 函数：打印错误
error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system() {
    info "检查系统要求..."
    
    # 检查Python版本
    MIN_PYTHON_VERSION="3.11"
    CURRENT_VERSION=$(python3 --version | cut -d ' ' -f 2)
    
    if [[ "$(printf '%s\n' "$MIN_PYTHON_VERSION" "$CURRENT_VERSION" | sort -V | head -n 1)" != "$MIN_PYTHON_VERSION" ]]; then
        error "Python版本至少需要 $MIN_PYTHON_VERSION，但当前版本是 $CURRENT_VERSION"
        return 1
    fi
    
    info "Python版本检查通过: $CURRENT_VERSION"
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        error "未找到pip3命令"
        return 1
    fi
    
    return 0
}

# 检查并创建虚拟环境
check_venv() {
    info "检查虚拟环境..."
    
    VENV_DIR="$PROJECT_ROOT/venv"
    
    if [ ! -d "$VENV_DIR" ]; then
        info "创建虚拟环境..."
        python3 -m venv "$VENV_DIR"
    fi
    
    info "激活虚拟环境..."
    source "$VENV_DIR/bin/activate"
    
    info "升级pip..."
    pip3 install --quiet --upgrade pip
    
    return 0
}

# 检查并安装依赖
check_dependencies() {
    info "检查依赖..."
    
    if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        error "未找到requirements.txt文件"
        return 1
    fi
    
    info "安装依赖..."
    pip3 install --quiet -r "$PROJECT_ROOT/requirements.txt"
    
    return 0
}

# 检查环境变量
check_env() {
    info "检查环境变量..."
    
    REQUIRED_VARS=("TUSHARE_TOKEN")
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!var}" ]; then
            if [ -f "$PROJECT_ROOT/.env" ]; then
                info "从 .env 文件读取环境变量..."
                while IFS= read -r line; do
                    if [[ "$line" == *"$var"* ]]; then
                        export "$line"
                        info "已加载 $var"
                    fi
                done < "$PROJECT_ROOT/.env"
            fi
            
            if [ -z "${!var}" ]; then
                error "缺少环境变量: $var"
                warn "请在 .env 文件中设置或直接导出到环境变量"
                return 1
            fi
        fi
    done
    
    # 验证TUSHARE token格式
    if [ ${#TUSHARE_TOKEN} -lt 10 ]; then
        error "TUSHARE_TOKEN格式不正确"
        return 1
    fi
    
    return 0
}

# 初始化数据库
init_database() {
    info "初始化数据库..."
    
    if [ ! -d "$PROJECT_ROOT/duckdb" ]; then
        mkdir -p "$PROJECT_ROOT/duckdb"
    fi
    
    info "执行数据库初始化..."
    python3 "$PROJECT_ROOT/scripts/db_init.py"
    
    return 0
}

# 导入示例数据
import_sample_data() {
    info "导入示例数据..."
    
    if [ -f "$PROJECT_ROOT/scripts/ingest_sample_data.py" ]; then
        info "执行示例数据导入..."
        python3 "$PROJECT_ROOT/scripts/ingest_sample_data.py"
    fi
    
    return 0
}

# 启动Streamlit服务器
start_streamlit() {
    info "启动Streamlit服务器..."
    
    STREAMLIT_CMD=(
        python3 -m streamlit run src/streamlit_app.py
        --server.port 8502
        --server.address 0.0.0.0
        --server.headless true
        --browser.gatherUsageStats false
        --logger.level info
    )
    
    info "服务器启动命令: ${STREAMLIT_CMD[*]}"
    
    # 启动服务器并记录日志
    nohup "${STREAMLIT_CMD[@]}" >> "$LOG_FILE" 2>&1 &
    STREAMLIT_PID=$!
    
    info "Streamlit服务器已启动，PID: $STREAMLIT_PID"
    echo $STREAMLIT_PID > "$PROJECT_ROOT/.streamlit_pid"
    
    # 等待服务器启动
    info "等待服务器启动..."
    sleep 5
    
    if ps -p $STREAMLIT_PID > /dev/null; then
        info "服务器启动成功！"
        info "访问地址: http://localhost:8502"
        info "外部访问: http://0.0.0.0:8502"
    else
        error "服务器启动失败，请检查日志: $LOG_FILE"
        return 1
    fi
    
    return 0
}

# 停止Streamlit服务器
stop_streamlit() {
    info "停止Streamlit服务器..."
    
    if [ -f "$PROJECT_ROOT/.streamlit_pid" ]; then
        STREAMLIT_PID=$(cat "$PROJECT_ROOT/.streamlit_pid")
        if kill -0 $STREAMLIT_PID 2>/dev/null; then
            info "正在停止服务器 (PID: $STREAMLIT_PID)..."
            kill $STREAMLIT_PID
            rm "$PROJECT_ROOT/.streamlit_pid"
            info "服务器已停止"
        else
            info "服务器已停止运行"
            rm "$PROJECT_ROOT/.streamlit_pid"
        fi
    else
        info "未找到运行中的服务器"
    fi
    
    return 0
}

# 检查服务状态
check_status() {
    info "检查服务状态..."
    
    if [ -f "$PROJECT_ROOT/.streamlit_pid" ]; then
        STREAMLIT_PID=$(cat "$PROJECT_ROOT/.streamlit_pid")
        if kill -0 $STREAMLIT_PID 2>/dev/null; then
            info "Streamlit服务器正在运行，PID: $STREAMLIT_PID"
            info "访问地址: http://localhost:8502"
            info "日志文件: $LOG_DIR"
        else
            warn "服务器PID文件存在但进程不存在"
            rm "$PROJECT_ROOT/.streamlit_pid"
        fi
    else
        info "Streamlit服务器未运行"
    fi
    
    return 0
}

# 显示帮助信息
show_help() {
    echo "Usage: $0 {start|stop|status|restart}"
    echo "  start    启动生产环境"
    echo "  stop     停止生产环境"
    echo "  status   检查服务状态"
    echo "  restart  重启生产环境"
    echo ""
    echo "生产环境配置:"
    echo "  - 服务端口: 8502"
    echo "  - 绑定地址: 0.0.0.0"
    echo "  - 日志目录: logs/"
    echo "  - 数据库: duckdb/valuation.duckdb"
}

# 主函数
main() {
    case "$1" in
        start)
            info "启动生产环境..."
            
            if ! check_system; then
                error "系统检查失败"
                return 1
            fi
            
            if ! check_venv; then
                error "虚拟环境检查失败"
                return 1
            fi
            
            if ! check_dependencies; then
                error "依赖检查失败"
                return 1
            fi
            
            if ! check_env; then
                error "环境变量检查失败"
                return 1
            fi
            
            if ! init_database; then
                error "数据库初始化失败"
                return 1
            fi
            
            if ! import_sample_data; then
                warn "示例数据导入失败，但系统仍可运行"
            fi
            
            if ! start_streamlit; then
                error "服务器启动失败"
                return 1
            fi
            
            info "生产环境启动成功！"
            info "访问地址: http://localhost:8502"
            info "日志文件: $LOG_FILE"
            
            return 0
            ;;
            
        stop)
            stop_streamlit
            return 0
            ;;
            
        status)
            check_status
            return 0
            ;;
            
        restart)
            info "重启生产环境..."
            stop_streamlit
            sleep 3
            start
            return 0
            ;;
            
        help|--help|-h)
            show_help
            return 0
            ;;
            
        *)
            error "未知命令: $1"
            show_help
            return 1
            ;;
    esac
}

# 记录执行信息
exec 3>&1 4>&2
exec 1>>"$LOG_FILE" 2>&1
{
    info "="*60
    info "A股绝对估值系统 - 生产环境启动"
    info "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    info "项目目录: $PROJECT_ROOT"
    info "日志文件: $LOG_FILE"
    info "="*60
    info ""
    
    # 执行主函数
    main "$@"
    EXIT_CODE=$?
    
    info ""
    info "执行完成，返回代码: $EXIT_CODE"
    info "="*60
    
} 3>&1 4>&2

# 恢复标准输出
exec 1>&3 2>&4
exec 3>&- 4>&-

# 返回执行结果
exit $EXIT_CODE
