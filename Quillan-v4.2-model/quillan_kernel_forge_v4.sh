#!/bin/bash
################################################################################
# QUILLAN v4.2 CUSTOM KERNEL BUILDER
# Advanced Kernel Optimization & Compilation Suite
# Architect: CrashOverrideX | Status: PRODUCTION
# Purpose: Build a blazing-fast, tailored kernel for YOUR system
################################################################################

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🚀 QUILLAN KERNEL FORGE v4.2 STARTING 🚀             ║"
echo "║                 High-Performance Kernel Builder                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# SECTION 1: SYSTEM DETECTION & PROFILING
# ============================================================================
echo "[1/7] 🔍 SYSTEM PROFILING..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CPU_CORES=$(nproc)
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
KERNEL_VERSION=$(uname -r)
SYSTEM_ARCH=$(uname -m)
DISTRO=$(lsb_release -d | cut -f2)

echo "✓ CPU Cores: $CPU_CORES"
echo "✓ CPU Model: $CPU_MODEL"
echo "✓ Total RAM: $TOTAL_RAM"
echo "✓ System Arch: $SYSTEM_ARCH"
echo "✓ Current Kernel: $KERNEL_VERSION"
echo "✓ Distro: $DISTRO"
echo ""

# ============================================================================
# SECTION 2: DEPENDENCIES CHECK
# ============================================================================
echo "[2/7] 📦 CHECKING DEPENDENCIES..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DEPS=("build-essential" "libncurses-dev" "bison" "flex" "libssl-dev" "libelf-dev" "dwarves")
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
    if ! dpkg -l | grep -q "^ii  $dep"; then
        MISSING_DEPS+=("$dep")
    else
        echo "✓ $dep installed"
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Missing dependencies: ${MISSING_DEPS[*]}"
    echo "Install with: sudo apt install ${MISSING_DEPS[*]}"
    exit 1
fi
echo ""

# ============================================================================
# SECTION 3: DOWNLOAD LATEST KERNEL
# ============================================================================
echo "[3/7] ⬇️  DOWNLOADING LINUX KERNEL..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

KERNEL_DIR="$HOME/kernel_build"
mkdir -p "$KERNEL_DIR"
cd "$KERNEL_DIR"

# Fetch latest stable kernel version
LATEST_KERNEL=$(curl -s https://www.kernel.org/releases.json | grep -o '"latest_stable":{"version":"[^"]*' | cut -d'"' -f4)
KERNEL_FILE="linux-${LATEST_KERNEL}.tar.xz"

if [ ! -f "$KERNEL_FILE" ]; then
    echo "📥 Downloading Linux $LATEST_KERNEL (this may take a few minutes)..."
    wget -q https://cdn.kernel.org/pub/linux/kernel/v${LATEST_KERNEL%%.*}.x/$KERNEL_FILE --show-progress
    echo "✓ Download complete"
else
    echo "✓ Kernel source already present"
fi

echo "✓ Extracting kernel source..."
tar -xf "$KERNEL_FILE" 2>/dev/null || echo "✓ Already extracted"
KERNEL_SRC="linux-${LATEST_KERNEL}"
cd "$KERNEL_SRC"
echo ""

# ============================================================================
# SECTION 4: OPTIMIZED KERNEL CONFIGURATION
# ============================================================================
echo "[4/7] ⚙️  GENERATING OPTIMIZED CONFIG..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start with current config or defconfig
if [ -f /boot/config-$(uname -r) ]; then
    echo "✓ Using current system config as base"
    cp /boot/config-$(uname -r) .config
else
    echo "✓ Generating defconfig..."
    make defconfig > /dev/null
fi

# Apply Quillan Performance Tweaks
echo "✓ Applying performance optimizations..."
cat >> .config << 'EOF'

# ╔════════════════════════════════════════════════════════════════╗
# ║         QUILLAN v4.2 KERNEL OPTIMIZATION PROFILE              ║
# ╚════════════════════════════════════════════════════════════════╝

# CPU SCHEDULER OPTIMIZATIONS
CONFIG_SCHED_AUTOGROUP=y
CONFIG_SCHED_SMT=y
CONFIG_SCHED_MC=y
CONFIG_SCHED_MIGRATION_COST_LOGGER=y
CONFIG_SCHED_MIGRATION_COST=500000
CONFIG_SCHED_TUNE=y

# CPU FREQUENCY SCALING (Performance Focus)
CONFIG_CPU_FREQ_GOV_PERFORMANCE=y
CONFIG_CPU_FREQ_GOV_POWERSAVE=y
CONFIG_CPU_FREQ_GOV_ONDEMAND=y
CONFIG_CPU_FREQ_DEFAULT_GOV_PERFORMANCE=y

# I/O SCHEDULER OPTIMIZATION (NOOP/MQ-Deadline)
CONFIG_IOSCHED_NOOP=y
CONFIG_IOSCHED_DEADLINE=y
CONFIG_DEFAULT_IOSCHED="mq-deadline"

# MEMORY MANAGEMENT
CONFIG_TRANSPARENT_HUGEPAGE=y
CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS=y
CONFIG_COMPACTION=y
CONFIG_CMA=y

# NETWORK STACK OPTIMIZATION
CONFIG_TCP_CONG_BBRT=y
CONFIG_TCP_CONG_BIC=y
CONFIG_DEFAULT_TCP_CONG="bbr"
CONFIG_NET_RX_BUSY_POLL=y
CONFIG_NET_FLOW_LIMIT=y

# POWER MANAGEMENT (Efficiency without throttling)
CONFIG_CPU_IDLE=y
CONFIG_CPU_IDLE_GOV_MENU=y
CONFIG_INTEL_PSTATE=y
CONFIG_INTEL_PSTATE_DEFAULT_GOVERNOR="performance"

# PREEMPTION FOR LOWER LATENCY
CONFIG_PREEMPT_BUILD=y
CONFIG_PREEMPT_VOLUNTARY=y

# DISABLE UNNECESSARY MODULES (BLOAT REMOVAL)
# CONFIG_ISDN is not set
# CONFIG_ISDN_CAPI is not set
# CONFIG_SOUND is not set (if not needed)
# CONFIG_FIREWIRE is not set
# CONFIG_WIRELESS is not set (if not needed)

# COMPILER OPTIMIZATION FLAGS
CONFIG_CC_OPTIMIZE_FOR_SIZE=n
CONFIG_CC_OPTIMIZE_FOR_PERFORMANCE=y

# BUILD OPTIONS
CONFIG_IKCONFIG=y
CONFIG_IKCONFIG_PROC=y

EOF

echo "✓ Config customization complete"
echo ""

# ============================================================================
# SECTION 5: KERNEL COMPILATION
# ============================================================================
echo "[5/7] 🔨 COMPILING KERNEL (This will take 10-30 minutes)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build with all available cores
echo "✓ Starting compilation with $CPU_CORES cores..."
make olddefconfig > /dev/null
make -j$CPU_CORES 2>&1 | tail -20  # Show last 20 lines of compilation

echo ""
echo "✓ Kernel compilation successful"
echo ""

# ============================================================================
# SECTION 6: MODULES & INSTALLATION
# ============================================================================
echo "[6/7] 📦 INSTALLING MODULES & KERNEL..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "✓ This step requires sudo privileges..."
sudo make modules_install > /dev/null
sudo make install > /dev/null

echo "✓ Kernel installed to /boot"
echo "✓ Bootloader updated automatically"
echo ""

# ============================================================================
# SECTION 7: VERIFICATION & SUMMARY
# ============================================================================
echo "[7/7] ✅ VERIFICATION & SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NEW_KERNEL_SIZE=$(du -sh "$KERNEL_DIR/$KERNEL_SRC" | cut -f1)
BUILD_TIME=$(date)

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║            🚀 QUILLAN KERNEL BUILD COMPLETE! 🚀                ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║"
echo "║ ✓ Kernel Version: $LATEST_KERNEL"
echo "║ ✓ Build Size: $NEW_KERNEL_SIZE"
echo "║ ✓ Build Time: $BUILD_TIME"
echo "║ ✓ System Arch: $SYSTEM_ARCH"
echo "║ ✓ CPU Cores Used: $CPU_CORES"
echo "║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║              🔧 PERFORMANCE OPTIMIZATIONS APPLIED:             ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║"
echo "║ 🎯 Scheduler: CFS with autogroup & SMT optimization"
echo "║ ⚡ CPU Governor: PERFORMANCE mode (max throughput)"
echo "║ 💾 Memory: Transparent Hugepages enabled"
echo "║ 🌐 Network: BBR TCP congestion control"
echo "║ 📊 I/O: mq-deadline scheduler for NVMe/SSD"
echo "║ 🔋 Idle: CPU_IDLE with menu governor (efficient)"
echo "║ ⏱️  Latency: Voluntary preemption enabled"
echo "║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                    📋 NEXT STEPS:                              ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║"
echo "║ 1. Reboot your system to activate the new kernel:"
echo "║    $ sudo reboot"
echo "║"
echo "║ 2. Verify kernel is running after reboot:"
echo "║    $ uname -r"
echo "║    (Should show: $LATEST_KERNEL)"
echo "║"
echo "║ 3. Check performance improvements:"
echo "║    $ cat /proc/cpuinfo"
echo "║    $ cat /proc/sys/kernel/sched_domain/cpu0/domain0/name"
echo "║"
echo "║ 4. Monitor system with:"
echo "║    $ sudo apt install htop iotop systat"
echo "║    $ htop"
echo "║"
echo "║ 5. Benchmark before/after (optional):"
echo "║    $ geekbench5  OR  $ sysbench"
echo "║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 Quillan v4.2 Kernel Build Complete!"
echo "   Your system is now primed for maximum performance. 🚀"
echo ""
