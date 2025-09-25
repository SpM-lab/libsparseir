using LibGit2

using HTTP
using JSON3
using Mustache

function get_latest_tag_commit(owner::String, repo::String)
    release_url = "https://api.github.com/repos/$owner/$repo/releases/latest"
    release_resp = HTTP.get(release_url)
    release_data = JSON3.read(String(release_resp.body))

    tag = release_data["tag_name"]  # e.g.: "v0.5.2"

    # 2. Get tag reference
    ref_url = "https://api.github.com/repos/$owner/$repo/git/ref/tags/$tag"
    ref_resp = HTTP.get(ref_url)
    ref_data = JSON3.read(String(ref_resp.body))

    object_sha = ref_data["object"]["sha"]
    object_type = ref_data["object"]["type"]

    # 3. For annotated tags, further resolve the tag object
    commit_sha = ""
    if object_type == "commit"
        commit_sha = object_sha
    elseif object_type == "tag"
        # For annotated tags, get the tag object
        tagobj_url = "https://api.github.com/repos/$owner/$repo/git/tags/$object_sha"
        tagobj_resp = HTTP.get(tagobj_url)
        tagobj_data = JSON3.read(String(tagobj_resp.body))
        commit_sha = tagobj_data["object"]["sha"]
    elseif object_type == "tree"
        commit_sha = object_sha
    else
        error("Unexpected object type: $object_type")
    end
    return (; tag, commit_sha)
end

function generate_libsparseir_build_tarballs(; version, commit_hash)
    template_path = joinpath(@__DIR__, "templates", "libsparseir.template")
    template = read(template_path, String)
    d = Dict(
        "libsparseir_version" => VersionNumber(version),
        "libsparseir_commit_hash" => commit_hash,
    )
    return Mustache.render(template, d)
end

function main()

    owner = "SpM-lab"
    repo = "libsparseir"
    yggdrasil_fork_repo_url = "git@github.com:SpM-lab/Yggdrasil.git"

    latest_tag_commit = get_latest_tag_commit(owner, repo)

    file_str = generate_libsparseir_build_tarballs(;
        version = latest_tag_commit.tag,
        commit_hash = latest_tag_commit.commit_sha,
    )
    write(joinpath(@__DIR__, "build_tarballs.jl"), file_str)

    yggdrasil_root = "Yggdrasil"
    if !isdir(yggdrasil_root)
        @info "Clone $(yggdrasil_root) from $(yggdrasil_fork_repo_url)"
        run(`git clone --depth=1 $(yggdrasil_fork_repo_url)`)
    end

    yggdrasil = LibGit2.GitRepo(yggdrasil_root)

    println("Cloned to: ", yggdrasil_root)
    path_to_build_tarballs = "L/libsparseir/build_tarballs.jl"
    cp("build_tarballs.jl", joinpath(yggdrasil_root, path_to_build_tarballs), force=true)
    branch_name = "libsparseir-$(latest_tag_commit.tag)"
    LibGit2.branch!(yggdrasil, branch_name)
    LibGit2.add!(yggdrasil, path_to_build_tarballs)
    LibGit2.commit(yggdrasil, "libsparseir: Update to version $(latest_tag_commit.tag)\n Update to version $(latest_tag_commit.tag)")
    run(`git -C $(yggdrasil_root) push --set-upstream origin $(branch_name)`)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

