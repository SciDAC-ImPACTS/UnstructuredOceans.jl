using Test
using Dates
using UnstructuredOceans: config_read, config_get, GlobalConfig, yaml_config,
            config_set, config_add

ref_hmix_String = "Restart_timestamp"
ref_hmix_Float  = 1.234567890
ref_hmix_None   = "none"
ref_hmix_On     = true
ref_hmix_Off    = false
ref_hmix_Exp    = 1.e25

# read the test configuration file in the "test/infra" folder
config = config_read(joinpath(@__DIR__, "test.yaml"))

# parse 
hmixConfig      = config_get(config.namelist, "hmix")
intervalsConfig = config_get(config.streams,  "intervals")
datetimesConfig = config_get(config.streams,  "datetimes")

# Test parsing various datatypes
@test ref_hmix_String == config_get(hmixConfig, "hmix_String") 
@test ref_hmix_Float  == config_get(hmixConfig, "hmix_Float") 
@test ref_hmix_None   == config_get(hmixConfig, "hmix_None") 
@test ref_hmix_On     == config_get(hmixConfig, "hmix_On") 
@test ref_hmix_Off    == config_get(hmixConfig, "hmix_Off") 
@test ref_hmix_Exp    == config_get(hmixConfig, "hmix_Exp") 

# Test parsing Periods
@test Year(1)   == config_get(intervalsConfig, "yearly_interval")
@test Month(2)  == config_get(intervalsConfig, "monthly_interval")
@test Day(3)    == config_get(intervalsConfig, "daily_interval")
@test Hour(4)   == config_get(intervalsConfig, "hourly_interval")
@test Minute(5) == config_get(intervalsConfig, "minutes_interval")
@test Second(6) == config_get(intervalsConfig, "seconds_interval")

# Test parsing DateTimes
@test DateTime(1,1,1,0,0,0) == config_get(datetimesConfig, "NO_HMS")
@test DateTime(1,1,1,2,0,0) == config_get(datetimesConfig, "NO_MS")  
@test DateTime(1,1,1,2,3,0) == config_get(datetimesConfig, "NO_S")
@test DateTime(1,1,1,0,3,4) == config_get(datetimesConfig, "NO_H")   
@test DateTime(1,1,1,0,0,4) == config_get(datetimesConfig, "NO_HM")  
@test DateTime(1,1,1,0,3,0) == config_get(datetimesConfig, "NO_HS")  
@test DateTime(1,1,1,2,3,4) == config_get(datetimesConfig, "ALL_HMS")
