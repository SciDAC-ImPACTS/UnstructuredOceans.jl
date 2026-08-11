using YAML

"""Making an alias for the type of dict attribute
to allow for more flexibility if it were to change 
in the future
"""
dict_type = Dict{Any, Any}

"""Types which inherit from `yaml_config` should have 
a `dict` attribute. 
"""
mutable struct yaml_config
    dict::dict_type
end

"""Default constructor for testing/debugging
"""
yaml_config() = yaml_config(dict_type())

"""
Struct which stores the namelist and streams config strucutres
"""
struct GlobalConfig
    namelist::yaml_config
    streams::yaml_config
end 

"""
Default constructor, only intended used for testing/debugging
"""
function GlobalConfig()
    GlobalConfig(yaml_config(dict_type()), yaml_config(dict_type()))
end

"""
Following Omega specs, define method to get

Allow the type info to be passed so that it 
can be used to create a new instance if string 
corresponds to header, not a config option. 
"""
function config_get(d::C, s::String) where {C<:yaml_config}
    
    # access the underlying dictionary contained within the object
    # and use key (string) to get config info 
    c = d.dict[s]
    
    # if a dictionary is returned that means we have reached the bottom 
    # level of the yaml tree, instead return a new instance of the 
    # configuration struct. 
    if typeof(c) == dict_type
        return C(c) 
    else
        return c
    end
end

""" Method for adding a new configuration option (and value)
"""
function config_add(d::C, s::String, val) where {C<:yaml_config}

    if haskey(d.dict, s)
        error("config_add: variable $(s) already exists use config_set instead")
    else
        d.dict[s] = val
    end
end

""" Method for overwriting value of existing configuration setting
"""
function config_set(d::C, s::String, val) where {C<:yaml_config}

    if haskey(d.dict, s)

        # check that the type of the new value is the same as existing
        if typeof(d.dict[s]) != typeof(val)
            @warn """config_set: Changing typeof \"$(s)\",
                     $(typeof(d.dict[s])) != $(typeof(val))
                  """
        end

        d.dict[s] = val
    else
        error("config_set: Could not find variable $(s)")
    end
end

#= DEFINE ALL THE SHARED METHODS HERE: 
    1. config get 
    2. config set (will require mutable struct) 
    3. config add (will require mutable struct)  
    4. config exists
    5. config write 
=# 

function config_read(filepath::AbstractString)

    # check that the config YAML file exists
    if !isfile(filepath)
        error("YAML configuration file does not exist")
    end
    
    # load YAML file as a dictionary
    config = YAML.load_file(filepath)

    # Extract the "streams"/"namelist" dicts from the global YAML dict
    streams = pop!(config["omega"], "streams")
    namelist = pop!(config, "omega")

    # Traverse the namelist/streams dicts and parse the timestamps/timeintervals
    streams  = parse_Datetimes(streams)
    namelist = parse_Datetimes(namelist)

    # Pack the namelist and streams into the global config struct 
    return GlobalConfig(yaml_config(namelist), yaml_config(streams))
end

function parse_Datetimes(dict::Dict{Any,Any})
    
    for (key, value) in dict

        # recursion to traverse nested dictionaries
        if value isa Dict
            parse_Datetimes(value)
            continue
        end

        # check if the timestamp pattern occurs in the string values
        if value isa String && occursin(timestamp_pat, value)
            dict[key] = datetime_from_string(value)
        end 
    end

    return dict
end

# to do: Need to generalize this regex pattern to work for all the possible 
#        MPAS timestamps options. 
const timestamp_pat = r"^(?:
                    (?:(\d{1,4})-)?      (?# year)
                    (?:(\d\d?)-)?        (?# month)
                    (\d+)                (?# day)
                    )?
                    _?                  
                    (\d\d):              (?# hour)
                    (\d\d):              (?# minute)
                    (\d\d)               (?# second)
                    $"x 

function index_to_period(captures)
    # get the non-zero index 
    idx = findall(x->!iszero(x), captures)[1]

    # https://github.com/JuliaLang/julia/issues/18285#issuecomment-1153218675
    idx == 1 && return Year(captures[idx])
    idx == 2 && return Month(captures[idx])
    idx == 3 && return Day(captures[idx])
    
    idx == 4 && return Hour(captures[idx])
    idx == 5 && return Minute(captures[idx])
    idx == 6 && return Second(captures[idx])
end

function datetime_from_string(string::String)

    mat = match(timestamp_pat, string)

    if mat === nothing
        throw(YAML.ConstructorError(nothing, nothing,
            "could not make sense of timestamp format", node.start_mark))
    end
    
    # If all fields are passed and non-zero, will be parsed as DateTime
    if all(mat.captures .!= nothing)
        h  = parse(Int, mat.captures[4])
        m  = parse(Int, mat.captures[5])
        s  = parse(Int, mat.captures[6])

        yr = parse(Int, mat.captures[1])
        mn = parse(Int, mat.captures[2])
        dy = parse(Int, mat.captures[3])
        
        # If month/day are equal to zero the string is a TimeInterval 
        # (c.f. DateTime) so the `index_to_period` method will do the parsing
        if !any(iszero.((mn,dy)))
            return DateTime(yr, mn, dy, h, m, s)
        end
    end 
    
    # if element is equal to nothing, return Int(0). Otherwise 
    # parse the string as an Int
    # https://stackoverflow.com/a/54393947
    captures = [x==nothing ? 0::Int : parse(Int,x) for x in mat.captures]
    
    # in the case where everything is zero or nothing, except one field 
    # return a period corresponding to that field 
    if count(!iszero, captures) == 1 
        return index_to_period(captures)
    end 
    
    h  = parse(Int, mat.captures[4])
    m  = parse(Int, mat.captures[5])
    s  = parse(Int, mat.captures[6])

    # No Y/M/D info in timestamp
    if all(mat.captures[1:3] .=== nothing)
        return Time(h, m, s)
    # When: "DDDD_HH:MM:SS" — always a duration; convert to total seconds
    elseif all(mat.captures[1:2] .=== nothing)
        days = parse(Int, mat.captures[3])
        return Second(days * 86400 + h * 3600 + m * 60 + s)
    end

    @warn """ Failed to parse $(string) """

    # in the case where we fail to parse the DateTime/Period just return 
    # the original string after having raised a warning
    return string
end