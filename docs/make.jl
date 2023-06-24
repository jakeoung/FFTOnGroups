using Documenter, DrWatson

@quickactivate "FFTOnGroups"

using Literate
using Plots # to not capture precompilation output


cd(projectdir()*"/docs")
cp("/Users/jakeoung/My Drive/docs/brain/2area/documenter/FFTOnGroups", "src/", force=true)

# generate examples
EXAMPLE = [
    joinpath(@__DIR__, "..", "scripts", "01_approx_cont_ft.jl"),
    # joinpath(@__DIR__, "..", "scripts", "02_test_SE2.jl")
]
OUTPUT = joinpath(@__DIR__, "src/generated")

# function preprocess(str)
#     str = replace(str, "x = 123" => "y = 321"; count=1)
#     return str
# end


# Literate.markdown(EXAMPLE, OUTPUT)
# Literate.notebook(EXAMPLE, OUTPUT)
# Literate.script(EXAMPLE, OUTPUT)

# Literate.markdown(joinpath(@__DIR__, "src/outputformats.jl"), OUTPUT; credit = false, name = "name")
# Literate.notebook(joinpath(@__DIR__, "src/outputformats.jl"), OUTPUT; name = "notebook")
# Literate.script(joinpath(@__DIR__, "src/outputformats.jl"), OUTPUT; credit = false)


#--------------------------------------------------
# generate docs
#--------------------------------------------------

makedocs(sitename="FFTOnGroups.jl",
    modules = [Literate],
    format = Documenter.HTML(prettyurls = false)
)



# EXAMPLE = joinpath(@__DIR__, "..", "scripts", "02_test_SE2.jl")
# OUTPUT = joinpath(@__DIR__, "src/generated")

# Literate.markdown(EXAMPLE, OUTPUT)
