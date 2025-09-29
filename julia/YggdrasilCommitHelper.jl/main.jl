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

function cleanup_work_branches(yggdrasil_root::String)
    @info "Cleaning up existing libsparseir work branches..."
    
    # Get all local branches
    try
        local_branches = read(`git -C $(yggdrasil_root) branch --format="%(refname:short)"`, String)
        libsparseir_branches = filter(branch -> occursin("libsparseir", branch), split(strip(local_branches), '\n'))
        
        # Delete local libsparseir branches
        for branch in libsparseir_branches
            if branch != "master"  # Don't delete master
                @info "Deleting local branch: $branch"
                try
                    run(`git -C $(yggdrasil_root) branch -D $branch`)
                catch e
                    @warn "Failed to delete local branch $branch: $e"
                end
            end
        end
    catch e
        @warn "Failed to list local branches: $e"
    end
    
    # Get all remote branches
    try
        remote_branches = read(`git -C $(yggdrasil_root) branch -r --format="%(refname:short)"`, String)
        libsparseir_remote_branches = filter(branch -> occursin("libsparseir", branch), split(strip(remote_branches), '\n'))
        
        # Delete remote libsparseir branches
        for branch in libsparseir_remote_branches
            if !occursin("master", branch)  # Don't delete master
                branch_name = replace(branch, "origin/" => "")
                @info "Deleting remote branch: $branch_name"
                try
                    run(`git -C $(yggdrasil_root) push origin --delete $branch_name`)
                    @info "Successfully deleted remote branch: $branch_name"
                catch e
                    @warn "Failed to delete remote branch $branch_name: $e"
                    # Try to delete the branch reference locally first
                    try
                        run(`git -C $(yggdrasil_root) branch -r -d $branch`)
                    catch e2
                        @warn "Failed to delete local reference to $branch: $e2"
                    end
                end
            end
        end
    catch e
        @warn "Failed to list remote branches: $e"
    end
    
    @info "Branch cleanup completed"
end

function sync_with_upstream(yggdrasil_root::String)
    @info "Syncing with upstream Yggdrasil repository..."
    
    # Checkout master branch
    @info "Checking out master branch..."
    run(`git -C $(yggdrasil_root) checkout master`)
    
    # Check if upstream remote exists
    try
        run(`git -C $(yggdrasil_root) remote get-url upstream`)
        @info "Upstream remote already exists"
    catch
        @info "Adding upstream remote..."
        run(`git -C $(yggdrasil_root) remote add upstream https://github.com/JuliaPackaging/Yggdrasil.git`)
    end
    
    # Fetch from upstream with depth limit for efficiency
    @info "Fetching from upstream..."
    try
        run(`git -C $(yggdrasil_root) fetch --depth=1000 upstream master`)
        @info "Successfully fetched from upstream"
    catch e
        @warn "Shallow fetch failed, attempting unshallow and full fetch: $e"
        # If shallow fetch fails, try to unshallow first
        try
            run(`git -C $(yggdrasil_root) fetch --unshallow`)
            run(`git -C $(yggdrasil_root) fetch upstream master`)
        catch e2
            @warn "Full fetch also failed: $e2"
            @warn "Skipping upstream sync due to fetch failures"
            return  # Skip sync if both fail
        end
    end
    
    # Reset to upstream/master to ensure we have the latest state
    @info "Resetting to upstream/master..."
    run(`git -C $(yggdrasil_root) reset --hard upstream/master`)
    @info "Successfully reset to upstream/master"
    
    # Clean up existing libsparseir work branches
    cleanup_work_branches(yggdrasil_root)
    
    # Fetch latest remote references to ensure we have up-to-date info
    @info "Fetching latest remote references..."
    run(`git -C $(yggdrasil_root) fetch --prune origin`)
    
    # Push updated master to origin
    @info "Pushing updated master to origin..."
    run(`git -C $(yggdrasil_root) push origin master`)
    
    @info "Upstream synchronization completed"
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

    yggdrasil_root = "Yggdrasil"
    if !isdir(yggdrasil_root)
        @info "Clone $(yggdrasil_root) from $(yggdrasil_fork_repo_url)"
        run(`git clone --depth=1000 $(yggdrasil_fork_repo_url)`)
    end

    yggdrasil = LibGit2.GitRepo(yggdrasil_root)
    
    # Ensure the repository is up-to-date with upstream
    sync_with_upstream(yggdrasil_root)

    println("Cloned to: ", yggdrasil_root)
    path_to_build_tarballs = "L/libsparseir/build_tarballs.jl"
    
    # Ensure the directory exists
    target_dir = dirname(joinpath(yggdrasil_root, path_to_build_tarballs))
    mkpath(target_dir)
    
    # Write the new build_tarballs.jl file directly (always replace with latest version)
    @info "Updating build_tarballs.jl to latest version..."
    write(joinpath(yggdrasil_root, path_to_build_tarballs), file_str)
    branch_name = "libsparseir-$(latest_tag_commit.tag)"
    LibGit2.branch!(yggdrasil, branch_name)
    LibGit2.add!(yggdrasil, path_to_build_tarballs)
    
    
    LibGit2.commit(yggdrasil, "libsparseir: Update to version $(latest_tag_commit.tag)\n Update to version $(latest_tag_commit.tag)")
    
    # Push the new branch, force push if it already exists
    @info "Pushing branch $branch_name to origin..."
    try
        run(`git -C $(yggdrasil_root) push --set-upstream origin $(branch_name)`)
        @info "Successfully pushed branch $branch_name"
    catch e
        @warn "Normal push failed, trying force-with-lease: $e"
        try
            run(`git -C $(yggdrasil_root) push --force-with-lease --set-upstream origin $(branch_name)`)
            @info "Successfully force-pushed branch $branch_name with --force-with-lease"
        catch e2
            @warn "Force-with-lease failed, trying regular force push: $e2"
            try
                run(`git -C $(yggdrasil_root) push --force --set-upstream origin $(branch_name)`)
                @info "Successfully force-pushed branch $branch_name with --force"
            catch e3
                @error "All push methods failed for branch $branch_name: $e3"
                throw(e3)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

