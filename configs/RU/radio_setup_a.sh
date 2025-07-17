#! /bin/sh
#================================================================
# HEADER
#================================================================
#% SYNOPSIS
#+    ${SCRIPT_NAME}
#%
#% DESCRIPTION
#%    This script is to initialize the RAN650 CAT A Benetel Radio Unit
#%
#% Note:
#%     Initialization sequence is as follows:
#%     1. Wait for sync service
#%     2. Initialize the RF-SOC
#%     3. Configure CFR and DPD
#%
#% OPTIONS
#%
#% EXAMPLES:
#%    ${SCRIPT_NAME}
#%
#================================================================
#- SCRIPT INFO
#-    version         ${SCRIPT_NAME} 1.2.2
#-    author          Benetel
#-    copyright       Copyright (c) Benetel
#-
#================================================================
#  HISTORY
#     19/01/2022 : AB : Script creation
#     23/02/2022 : IS : Additional PRACH register write for v0.4.7 + Uplink VLAN tag
#     02/03/2022 : AB : Add static Phase compensation for 4145.01 MHz
#     02/03/2022 : AB : Replace static Phase compensation for dynamic phase compensation
#     22/03/2022 : IS : v0.6: Introduces attenuation passed by argument, removes eeprom recovery as this is handled in separate service.
#     04/04/2022 : IS : v0.6: Merge Mavenir changes from v0.4.8 release.
#     26/04/2022 : IS : v0.6: Update logging output w.r.t FPGA register writes
#     13/05/2022 : SW : v0.6: Updated the Radiocontrol Version check and added a write to the Reg 0xC0322
#     23/06/2022 : IS : v0.6.1: Configure eAxCID
#     05/07/2022 : IS : v0.6.2: Configure eAxCID (based on /etc/eaxc.xml)
#     22/07/2022 : IS : v0.6.4: TDD switch alignment register addition
#     05/08/2022 : IS : v0.6.4: TDD switch alignment register edit
#     27/09/2022 : IS : v0.8.0 beta: configured for 4x2 for the beta release & implement DPD reset fix
#     23/01/2023 : KR : v0.8.1 Fixed the TAE alignment and compression support for short form prach 
#     08/05/2023 : RK : v1.0.0 RAN550 & 650 combined and -1 & -3 options also. Configured the Compression registers as needed for all modes
#     16/06/2023 : RK&LR : v1.0.0 Modifications to config file and startup script for LF and SF prach compression settings
#     31/10/2023 : GU : v1.2.1 Added workaround for the BUG 7512 (TX accasionaly not transmitting)
#     20/11/2023 : GU : v1.2.2 Modified the startup procedure (DL mode prior to DPD, TDD mode only after DPD,
#                              PA Protection before CFR), BUG 7919 workaround.
#
#================================================================
#  DEBUG OPTION
#    set -n  # Uncomment to check your syntax, without execution.
#    set -x  # Uncomment to debug this shell script
#
#================================================================
# END_OF_HEADER
#================================================================
#== needed variables ==#
SCRIPT_HEADSIZE=$(head -n 200 ${0} |grep -n "^# END_OF_HEADER" | cut -f1 -d:)
SCRIPT_NAME="$(basename ${0})"
SRC_PATH=${PWD}

#== usage functions ==#
usage() { printf "Usage: "; head -n ${SCRIPT_HEADSIZE:-99} ${0} | grep -e "^#+" | sed -e "s/^#+[ ]*//g" -e "s/\${SCRIPT_NAME}/${SCRIPT_NAME}/g" ; }
usagefull() { head -n ${SCRIPT_HEADSIZE:-99} ${0} | grep -e "^#[%+-]" | sed -e "s/^#[%+-]//g" -e "s/\${SCRIPT_NAME}/${SCRIPT_NAME}/g" ; }
scriptinfo() { head -n ${SCRIPT_HEADSIZE:-99} ${0} | grep -e "^#-" | sed -e "s/^#-//g" -e "s/\${SCRIPT_NAME}/${SCRIPT_NAME}/g"; }
## Globals
LOG_ROOT=/tmp/logs
LOG_STATUS_FP=${LOG_ROOT}/radio_status
LOG_RAD_STAT_FP=${LOG_ROOT}/radio_boot_response
SCRIPT_ROOT=/usr/sbin
FEM_RESET_REGISTER="0xC0372"
FEM_RESET_BIT=0

LOG_INFO() {
    echo "[INFO] $1" >> ${LOG_STATUS_FP}
}
LOG_WARN() {
    echo "[WARNING] $1" >> ${LOG_STATUS_FP}
}
LOG_DEBUG() {
    echo "[DEBUG] $1" >> ${LOG_STATUS_FP}
}
LOG_ERROR() {
    echo "[ERROR] $1" >> ${LOG_STATUS_FP}
}

RADIOCONTROL_COMPAT_PROD=MP15
RADIOCONTROL_COMPAT_VERSION=1.1.0

version_check()
{
    COMPAT_PROD=$( /usr/bin/radiocontrol -V | grep "Product ID: $RADIOCONTROL_COMPAT_PROD")
    if [ $? -ne 0 ]
    then
        return $?
    fi

    VERSION=$( /usr/bin/radiocontrol -V | grep "Version: $RADIOCONTROL_COMPAT_VERSION")
    return $?
}

SYNC_STATE_LOCKED_STRING="0"
GPS_MODE_STRING=GPS
PTP_MODE_STRING=PTP
FREERUN_MODE_STRING=FREERUN
RU_SYNC_MODE_FILEPATH=/etc/ru-sync-mode
MPLANE_ENABLED_STRING="enabled"
scriptinfo

isMplaneEnabled()
{
        mplaneEnableStatus=$(systemctl is-enabled mplane)

        if [ $mplaneEnableStatus = $MPLANE_ENABLED_STRING ]; then
                #echo "M-Plane is enabled"
                return 1
        else
                #echo "M-Plane is not enabled"
                return 0
        fi
}

tddinit()
{
    tdd_pattern=$(cat /etc/ru_cal_pattern.xml | grep description | cut -d '>' -f 2 | cut -d '<' -f 1)
    LOG_INFO "Setting initialization TDD Pattern: $tdd_pattern"
    /usr/sbin/initTddConfig "/etc/ru_cal_pattern.xml" >> ${LOG_RAD_STAT_FP}
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi
}

radioinit()
{
     # Enable PA protection for all products
    LOG_INFO "Enable PA protection on transmitters"
    /usr/bin/radiocontrol -o P e 15 >> ${LOG_RAD_STAT_FP}
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi
    sleep 3
    
    # load GP interrupt configuration into Madura
    LOG_INFO "Load GP interrupt configuration into Madura"
    /usr/bin/radiocontrol -o L g >> ${LOG_RAD_STAT_FP}
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi
    sleep 2

    # Call interrupt handler once to clear any sticky interrupts left over from initialization
    LOG_INFO "Call interrupt handler once to clear any sticky interrupts left over from initialization"
    /usr/bin/radiocontrol -o L h 1 >> ${LOG_RAD_STAT_FP}
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi

    if [ $mimo_mode == "1_3" ] || [ $mimo_mode == "2_4" ]
    then
        cfr_setting=0.47
    else
        cfr_setting=0.54
    fi

    if [ $mimo_mode == "1_3" ] || [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
    then
        LOG_INFO "Configure CFR for Antenna 1 ($cfr_setting)"
        /usr/bin/radiocontrol -o C c 1 $cfr_setting n >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3
    fi

    if [ $mimo_mode == "2_4" ] || [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
    then
        LOG_INFO "Configure CFR for Antenna 2 ($cfr_setting)"
        /usr/bin/radiocontrol -o C c 2 $cfr_setting n >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3
    fi

    if [ $mimo_mode == "1_3" ] || [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
    then
        LOG_INFO "Configure CFR for Antenna 3 ($cfr_setting)"
        /usr/bin/radiocontrol -o C c 4 $cfr_setting n >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3
    fi

    if [ $mimo_mode == "2_4" ] || [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
    then
        LOG_INFO "Configure CFR for Antenna 4 ($cfr_setting)"
        /usr/bin/radiocontrol -o C c 8 $cfr_setting n >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3
    fi

    LOG_INFO "Enable FPGA-controlled TDD switching"
    /usr/bin/radiocontrol -o M T >> ${LOG_RAD_STAT_FP}
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi
    sleep 3
    /usr/bin/radiocontrol -o F 2 >> ${LOG_RAD_STAT_FP}
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi
    sleep 3

    if [ $hardware_version == "RAN650" ]
    then
        LOG_INFO "Enable TX on FEM"
        /usr/bin/radiocontrol -o E 12 >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3

        LOG_INFO "FEM to $mimo_mode MIMO mode"
        if [ $mimo_mode == "1_3" ]
        then
            /usr/bin/radiocontrol -o E 9 >> ${LOG_RAD_STAT_FP}
        elif [ $mimo_mode == "2_4" ]
        then
            /usr/bin/radiocontrol -o E 8 >> ${LOG_RAD_STAT_FP}
        elif [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
        then
            /usr/bin/radiocontrol -o E 2 >> ${LOG_RAD_STAT_FP}
        else
            LOG_ERROR "mimo_mode not set in /etc/ru_config.cfg"
            exit 1
        fi
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 10
    fi

    if [ $mimo_mode == "1_3" ] || [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
    then
        LOG_INFO "DPD Tx1 configuration"
        /usr/bin/radiocontrol -o D c 1 n y >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3
    fi

    if [ $mimo_mode == "2_4" ] || [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
    then
        LOG_INFO "DPD Tx2 configuration"
        /usr/bin/radiocontrol -o D c 2 n y >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3
    fi

    if [ $mimo_mode == "1_3" ] || [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
    then
        LOG_INFO "DPD Tx3 configuration"
        /usr/bin/radiocontrol -o D c 4 n y >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3
    fi

    if [ $mimo_mode == "2_4" ] || [ $mimo_mode == "1_2_3_4_4x2" ] ||[ $mimo_mode == "1_2_3_4_4x4" ]
    then
        LOG_INFO "DPD Tx4 configuration"
        /usr/bin/radiocontrol -o D c 8 n y >> ${LOG_RAD_STAT_FP}
        if [ $? -ne 0 ]
        then
            LOG_ERROR "Failed with $?"
            exit 1
        fi
        sleep 3
    fi
    
    #Timeoffset register for TA adjustment
    ####LOG_INFO "Configuring the timeOffset register for TA adjustment"
    #####registercontrol -w c0304 -x 0xB0000001 >> ${LOG_RAD_STAT_FP}
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi
}

version_check
if [ $? -ne 0 ]
then
    LOG_ERROR "Only works with radiocontrol ${RADIOCONTROL_COMPAT_PROD}-${RADIOCONTROL_COMPAT_VERSION}"
fi

# Read the config file
. /etc/ru_config.cfg
if [ $? -ne 0 ]
then
    LOG_ERROR "Error reading ru_config.cfg"
    exit 1
fi

#Avoid corrupt radio control file from previous boot being used.
[ -f "/home/root/dev_struct.dat" ] && rm /home/root/dev_struct.dat || :

#Avoid false reads of files from last boot.
[ -f "/var/syncmon/sync-state" ] && rm /var/syncmon/sync-state || :

#Initialize TDD Pattern
LOG_INFO "Initialize TDD Pattern"
tddinit

#Assume PTP mode by default but check settings file for alternative mode
syncmode=PTP
if [ -f $RU_SYNC_MODE_FILEPATH ]; then
        syncmode=`cat $RU_SYNC_MODE_FILEPATH`
fi

if [ $syncmode = $FREERUN_MODE_STRING ]; then
    LOG_ERROR "RU set to FREERUN. Skipping check for Sync lock."
else
    ### STEP 1: Wait 3 mins for Sync
    sync_mode=$(cat /etc/ru-sync-mode)
    LOG_INFO "Waiting for $sync_mode sync"
    # TODO print sync mode
    # Wait for sync to be achieved
    while [ 1 ]
    do
        if [ -f /var/syncmon/sync-state ]
        then
            syncstate=`cat /var/syncmon/sync-state`
            if [ $syncstate = $SYNC_STATE_LOCKED_STRING ]; then
                sleep 30
                LOG_INFO "Sync completed"
                break
            fi
        fi
        sleep 1
        count=$((count+1))

        #if tod doesn't sync in 3 minutes, something is wrong. Exit.

        if [[ $count -gt 180 ]]; then
            LOG_ERROR "RU did not synchronize within 3 mins. Exiting radio initialization." >> ${LOG_RAD_STAT_FP}
            exit 1
        fi

    done
fi

### STEP 2: Change the CSR Registers and then Complete the remaining Radio config via RadioInit() function ###

LOG_INFO "Disable transmission during radio initialisation"
/usr/bin/registercontrol -w 0xc0300 -x 0x0

LOG_INFO "Set expected DU MAC Address for C-Plane Traffic (C0319/C031A)"
/usr/bin/registercontrol -w C031A -x 0xC84B
/usr/bin/registercontrol -w C0319 -x 0xD69D5B1F

LOG_INFO "Set expected DU MAC Address for U-Plane Traffic (C0315/C0316)"
/usr/bin/registercontrol -w C0316 -x 0xC84B
/usr/bin/registercontrol -w C0315 -x 0xD69D5B1F

LOG_INFO "Set required DU VLAN Tag Control Information for uplink U-Plane Traffic (C0318)"
/usr/bin/registercontrol -w C0318 -x 0x5

LOG_INFO "Set expected DU VLAN Tag Control Information for downlink U-Plane Traffic (C0330)"
/usr/bin/registercontrol -w C0330 -x 0x5

LOG_INFO "Set expected DU VLAN Tag Control Information for downlink C-Plane Traffic (C0331)"
/usr/bin/registercontrol -w C0331 -x 0x5

LOG_INFO "Aligning FPGA uplink timing to arrival of uplink frame(C0303)"
/usr/bin/registercontrol -w c0303 -x 0x20E

#### Check to see if benetel_extend_eeprom.sh was run #####
extend_eeprom_check=$(eeprog_v2 -b 0:0x57 -x -r 0x538:1 | tr -d ' ')
if [ $extend_eeprom_check == "ff" ]  || [ $extend_eeprom_check == "FF" ]
then
    benetel_extend_eeprom.sh
    LOG_ERROR "The setting is incorrect and benetel_extend_eeprom script was executed"
    LOG_ERROR "Please reboot the O-RU"
    exit 1
else
    LOG_INFO "The benetel_extend_eeprom check is complete and passed"
fi

if [ $downlink_scaling == "0" ] || [ $downlink_scaling == "6" ] || [ $downlink_scaling == "12" ] || [ $downlink_scaling == "18" ]; then
    LOG_INFO "Set Downlink scaling to $downlink_scaling dB (C0358)"
    /usr/bin/registercontrol -w C0358 -x 0x$downlink_scaling
else
    LOG_ERROR "downlink_scaling not set correctly in /etc/ru_config.cfg, exiting. Allowed values are 0, 6, 12, and 18 dB"
    exit 1
fi

##### To Enable the short form Prach c0328, c0329
if [ $prach_format == "short" ]; then
    LOG_INFO "Enable short form PRACH (C0328)"
    /usr/bin/registercontrol -w c0328 -x 0x3
elif [ $prach_format == "long" ]; then
    LOG_INFO "Enable long form PRACH (C0328)"
    /usr/bin/registercontrol -w c0328 -x 0x1
else
    LOG_ERROR "prach format not set in /etc/ru_config.cfg, exiting"
    exit 1
fi

if [ $compression == "static_uncompressed" ]; then
    if [ $mimo_mode == "1_2_3_4_4x2" ] || [ $mimo_mode == "1_2_3_4_4x4" ]; then
        LOG_ERROR "Uncompressed mode not valid for 4x2 or 4x4 modes, check /etc/ru_config.cfg"
        exit 1
    fi
    LOG_INFO "Compression mode: Static, Compression method: Uncompressed "
    /usr/bin/registercontrol -w C0350 -x 0x0
    /usr/bin/registercontrol -w C0351 -x 0x0
    /usr/bin/registercontrol -w C0352 -x 0x0
    /usr/bin/registercontrol -w C0354 -x 0x0
    /usr/bin/registercontrol -w C037C -x 0x0
elif [ $compression == "static_compressed" ]; then
    LOG_INFO "Compression mode: Static, Compression method: 9-bit BFP "
    /usr/bin/registercontrol -w C0350 -x 0x0
    /usr/bin/registercontrol -w C0351 -x 0x0
    /usr/bin/registercontrol -w C0352 -x 0x0
    /usr/bin/registercontrol -w C0354 -x 0x91
    /usr/bin/registercontrol -w C037C -x 0x9
elif [ $compression == "dynamic_compressed" ]; then
    LOG_INFO "Compression mode: Dynamic (only 9-bit BFP currently supported) "
    /usr/bin/registercontrol -w C0350 -x 0x1
    /usr/bin/registercontrol -w C0351 -x 0x1
    /usr/bin/registercontrol -w C0352 -x 0x1
    /usr/bin/registercontrol -w C0354 -x 0xFF
    /usr/bin/registercontrol -w C037C -x 0x9
elif [ $compression == "dynamic_uncompressed" ]; then
    if [ $mimo_mode == "1_2_3_4_4x2" ] || [ $mimo_mode == "1_2_3_4_4x4" ]; then
        LOG_ERROR "Uncompressed mode not valid for 4x2 or 4x4 modes, check /etc/ru_config.cfg"
        exit 1
    fi
    LOG_INFO "Compression mode: Dynamic, Uncompressed (FlexRAN bug workaround: udCompHdr enabled, uncompressed data)"
    /usr/bin/registercontrol -w C0350 -x 0x1
    /usr/bin/registercontrol -w C0351 -x 0x1
    /usr/bin/registercontrol -w C0352 -x 0x1
    /usr/bin/registercontrol -w C0354 -x 0x8
    /usr/bin/registercontrol -w C037C -x 0x0
else
    LOG_WARN "compression not set in /etc/ru_config.cfg, using uncompressed mode"
    /usr/bin/registercontrol -w C0354 -x 0x0
fi

hardware_version=$(cat /etc/manifest.xml | grep "bldVersion" | cut -d '"' -f2 | cut -c1-6)

release_version=$(cat /etc/manifest.xml | grep "bldVersion" | cut -d '"' -f2 | tail -c 2)
if [ $release_version == "1" ]; then
    LOG_INFO "Aligning TDD switching relative to downlink and uplink data and with respect to PPS (C0366)"
    /usr/bin/registercontrol -w C0366 -x 0x1500

    LOG_INFO "Set expected RU PRACH Configuration Index (C0322)"
    /usr/bin/registercontrol -w C0322 -x 0xE

    # This is controlled by set_bandwidth for now (FlexRAN 21.11 bug workaround)
    # LOG_INFO "Set short form PRACH FreqOffset (C0329)"
    # /usr/bin/registercontrol -w c0329 -x 0xFFF334

elif [ $release_version == "3" ]; then
    LOG_INFO "Aligning TDD switching relative to downlink and uplink data and with respect to PPS (C0366)"
    /usr/bin/registercontrol -w C0366 -x 0x1500

    LOG_INFO "Set expected RU PRACH Configuration Index (C0322)"
    /usr/bin/registercontrol -w C0322 -x 0x7
else
    LOG_ERROR "Release version $release_version not valid, check manifest file"
    exit 1
fi

if [ $cplane_per_symbol_workaround == "disabled" ] && [ $cuplane_dl_coupling_sectionID == "disabled" ]; then
    LOG_INFO "C-plane per symbol disabled and cuplane dl coupling via section ID disabled (C037B to 0x1)"
    /usr/bin/registercontrol -w C037B -x 0x1
elif [ $cplane_per_symbol_workaround == "disabled" ] && [ $cuplane_dl_coupling_sectionID == "enabled" ]; then
    LOG_INFO "C-plane per symbol disabled and cuplane dl coupling via section ID enabled (C037B to 0x0)"
    /usr/bin/registercontrol -w C037B -x 0x0
elif [ $cplane_per_symbol_workaround == "enabled" ] && [ $cuplane_dl_coupling_sectionID == "disabled" ]; then
    LOG_INFO "C-plane per symbol disabled and cuplane dl coupling via section ID disabled (C037B to 0x3)"
    /usr/bin/registercontrol -w C037B -x 0x3
elif [ $cplane_per_symbol_workaround == "enabled" ] && [ $cuplane_dl_coupling_sectionID == "enabled" ]; then
    LOG_INFO "C-plane per symbol disabled and cuplane dl coupling via section ID disabled (C037B to 0x2)"
    /usr/bin/registercontrol -w C037B -x 0x2
else
    LOG_ERROR "Please review /etc/ru_config.cfg configuration settings for cplane_per_symbol_workaround and cuplane_dl_coupling_sectionID"
    exit 1
fi

#if [ $cplane_per_symbol_workaround == "disabled" ]; then
#    LOG_INFO "C-plane per symbol workaround not enabled (C037B to 0x0)"
#    /usr/bin/registercontrol -w C037B -x 0x0
#elif [ $cplane_per_symbol_workaround == "enabled" ]; then
#    LOG_INFO "Enabling C-plane per symbol workaround (C037B to 0x1)"
#    /usr/bin/registercontrol -w C037B -x 0x1
#else
#    LOG_WARN "cplane_per_symbol_workaround not set in /etc/ru_config.cfg, setting to disabled"
#    /usr/bin/registercontrol -w C037B -x 0x0
#fi

if [ $lf_prach_compression_enable == "true" ]; then
    LOG_INFO "enable the LF prach compression (C0353)"
    /usr/bin/registercontrol -w c0353 -x 0x1
elif [ $lf_prach_compression_enable == "false" ]; then
    LOG_INFO "disable the LF prach compression (C0353)"
    /usr/bin/registercontrol -w c0353 -x 0x0
else
    LOG_ERROR "lf_prach_compression_enable not set in /etc/ru_config.cfg, exiting"
    exit 1
fi

if [ $flexran_prach_workaround == "enabled" ]; then
    LOG_INFO "FlexRAN Workaround: Uncompressed prach with udCompHdr enabled"
    /usr/bin/registercontrol -w c0354 -x 0x8
fi

if [ $mimo_mode == "1_3" ]
then
    LOG_INFO "Enabling 2 SF PRACH receive paths"
    /usr/bin/registercontrol -w c032a -x 0x3
elif [ $mimo_mode == "2_4" ]
then
    LOG_INFO "Enabling 2 SF PRACH receive paths"
    /usr/bin/registercontrol -w c032a -x 0x3
elif [ $mimo_mode == "1_2_3_4_4x2" ]
then
    LOG_INFO "Enabling 2 SF PRACH receive paths"
    /usr/bin/registercontrol -w c032a -x 0x3
elif [ $mimo_mode == "1_2_3_4_4x4" ]
then
    LOG_INFO "Enabling 4 SF PRACH receive paths"
    /usr/bin/registercontrol -w c032a -x 0xF
else
    LOG_ERROR "mimo_mode not set correctly in /etc/ru_config.cfg, exiting"
    exit 1
fi

isMplaneEnabled
mplaneStatus=$?

if [ $mplaneStatus -eq 0 ]
then
        LOG_INFO "Configure eAxC ID via /etc/eaxc.xml settings, as M-Plane is disabled."
        if [ $release_version == "1" ]; then
            if [ $mimo_mode == "1_3" ] || [ $mimo_mode == "2_4" ]; then
                cp /etc/eaxc_1_2t.xml /etc/eaxc.xml
            else
                cp /etc/eaxc_1_4t.xml /etc/eaxc.xml
            fi
        elif [ $release_version == "3" ]; then
            if [ $mimo_mode == "1_3" ] || [ $mimo_mode == "2_4" ]; then
                cp /etc/eaxc_3_2t.xml /etc/eaxc.xml
            else
                cp /etc/eaxc_3_4t.xml /etc/eaxc.xml
            fi
        fi
        initEaxcConfig
fi

if [ $hardware_version == "RAN650" ]
then
    ### Set the FEM_RESET_B signal to wake up the FEM
    fem_reset_value=$(($(registercontrol -r $FEM_RESET_REGISTER | awk '{print $5}')))
    #echo "READBACK: " "$fem_reset_value"
    fem_reset_value=$(($fem_reset_value | $((1<<$FEM_RESET_BIT))))
    #echo "Bit 0 Set: " "$fem_reset_value"
    /usr/bin/registercontrol -w $FEM_RESET_REGISTER  -d "$fem_reset_value"
    fem_reset_value=$(($(registercontrol -r $FEM_RESET_REGISTER | awk '{print $5}')))
    #echo "READBACK after setting Bit 0: " "$fem_reset_value"
fi

### STEP 3: Initialize the Madura

cd /home/root
LOG_INFO "Start Radio Configuration"
phase_comp -f $1

if [ $mimo_mode == "1_3" ]
then
    /usr/bin/radiocontrol -o I $1 2 255 >> ${LOG_RAD_STAT_FP}
elif [ $mimo_mode == "2_4" ]
then
    /usr/bin/radiocontrol -o I $1 3 255 >> ${LOG_RAD_STAT_FP}
elif [ $mimo_mode == "1_2_3_4_4x2" ]
then
    /usr/bin/radiocontrol -o I $1 4 255 >> ${LOG_RAD_STAT_FP}
elif [ $mimo_mode == "1_2_3_4_4x4" ]
then
    /usr/bin/radiocontrol -o I $1 6 255 >> ${LOG_RAD_STAT_FP}
else
    LOG_ERROR "mimo_mode not set in /etc/ru_config.cfg"
    exit 1
fi
if [ $? -ne 0 ]
then
    LOG_ERROR "Failed with $?"
    exit 1
fi
LOG_INFO "Initialize RF SoC"
LOG_INFO "Center frequency set to $1 MHz"

status=$?

if [ $status -eq 0 ]
then
    LOG_INFO "Initialize RF SoC"
    radioinit
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Radio Init failed with $?"
        exit 1
    fi

    if [ -f /etc/ru-tx-gain-db ]
    then
        GAIN_SETTING=$(cat /etc/ru-tx-gain-db)
    else
        GAIN_SETTING=$(cat /tmp/logs/ru_information | grep "Max Output Power - MCH" | tail -c6)
    fi

    #Bug7919 workaround
    LOG_INFO "Disable transmission"
    /usr/bin/registercontrol -w 0xc0300 -x 0x0

    LOG_INFO "Set the gain to $GAIN_SETTING dB"
    gaincontrol -g $GAIN_SETTING
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi

    #Bug7919 workaround
    LOG_INFO "Disable transmission"
    /usr/bin/registercontrol -w 0xc0300 -x 0x0

    if [ -f /etc/ru-bandwidth ]
    then
        BANDWIDTH_SETTING=$(cat /etc/ru-bandwidth)
    else
        BANDWIDTH_SETTING=100000000
    fi

    LOG_INFO "Set the bandwidth to $BANDWIDTH_SETTING Hz"
    set_bandwidth -b $BANDWIDTH_SETTING
    if [ $? -ne 0 ]
    then
        LOG_ERROR "Failed with $?"
        exit 1
    fi
    
    LOG_INFO "Modifying the TDD pattern from default to custom"
    sleep 2
    ## This will load the tdd pattern from /etc/tdd.xml at the end of the RU configuration
    initTddConfig
    sleep 2
    
    # To avoid set_bandwidth hardcoding the FreqOffset modify the register c0329 to read the FreqOffset from C-Plane
    # LOG_INFO "Set short form PRACH FreqOffset (C0329) to read it from C-Plane"
    # /usr/bin/registercontrol -w c0329 -x 0xFFF000

    if [ $mimo_mode == "1_3" ] || [ $mimo_mode == "2_4" ]
    then
        /usr/bin/registercontrol -w 0xc0300 -x 0x101 >> ${LOG_RAD_STAT_FP}
        /usr/bin/registercontrol -w 0xc0302 -x 0x101 >> ${LOG_RAD_STAT_FP}
        LOG_INFO "Transmission enabled (2x2)"
    elif [ $mimo_mode == "1_2_3_4_4x2" ]
    then
        /usr/bin/registercontrol -w 0xc0300 -x 0x1010101 >> ${LOG_RAD_STAT_FP}
        /usr/bin/registercontrol -w 0xc0302 -x 0x101 >> ${LOG_RAD_STAT_FP}
        LOG_INFO "Transmission enabled (4x2)"
    elif [ $mimo_mode == "1_2_3_4_4x4" ]
    then
        /usr/bin/registercontrol -w 0xc0300 -x 0x1010101 >> ${LOG_RAD_STAT_FP}
        /usr/bin/registercontrol -w 0xc0302 -x 0x1010101 >> ${LOG_RAD_STAT_FP}
        LOG_INFO "Transmission enabled (4x4)"
    else
        LOG_ERROR "mimo_mode not set in /etc/ru_config.cfg"
        exit 1
    fi

    if [ $dl_tuning_special_slot ]
    then
        LOG_INFO "Setting DL UL tuning special slot to $dl_tuning_special_slot"
        /usr/bin/registercontrol -w 0xc031F -x $dl_tuning_special_slot >> ${LOG_RAD_STAT_FP}
    fi

: '
    #GU BUGFIX: 7512
    #TX2 Bug7512 workaroung works only for 100MHz due to CW test limitation
    if [ $BANDWIDTH_SETTING -eq 100000000 ]; then
       DIG_ATT_REGISTER="0xC0301"
        RFSOC_NROF_INITS_MAX=3
        current_digital_gain=$(($(registercontrol -r $DIG_ATT_REGISTER | awk '{print $5}')))

        for i in `seq 1 $RFSOC_NROF_INITS_MAX`
        do
           /usr/sbin/check_tx.sh; TXOK=$?
           if [ $TXOK -eq 1 ]; then
               echo "All TX streams OK"
               break
           else
               #If unsuccesful then continue
               if [ $i -eq $RFSOC_NROF_INITS_MAX ]; then
                   echo "RF SoC init failure. Nrof. reinitializations reached max=$RFSOC_NROF_INITS_MAX. RU bring up stopped!!!"
                   LOG_ERROR "RF SoC init failure. Nrof. reinitializations reached max=$RFSOC_NROF_INITS_MAX. RU bring up stopped!!!"
                   break
               fi
           echo "RF SoC Reinitialization. Iteration $i"
           /usr/sbin/rf_soc_reinit_a.sh
           fi
        done
        #echo "Apply the gain [dec]: $current_digital_gain"
        registercontrol -w $DIG_ATT_REGISTER -d $current_digital_gain
    fi
'
    exit 0
else
    exit 253
fi