# frozen_string_literal: true

require "fileutils"
require "rbconfig"

require_relative "gem_progress"

module Mvcc
  # `gem install mvcc` runs this from ext/mvcc/extconf.rb: build the toolkit
  # from the sources packed in the gem, then add toolkit/bin to shell profile(s).
  module Installer
    MARKER_BEGIN = "# >>> mvcc toolkit >>>"
    MARKER_END = "# <<< mvcc toolkit <<<"

    module_function

    def gem_root
      File.expand_path("../..", __dir__)
    end

    def bindir
      File.join(gem_root, "toolkit", "bin")
    end

    def export_line(path = bindir)
      %(export PATH="#{path}:$PATH")
    end

    def fish_path_line(path = bindir)
      "fish_add_path -m --prepend #{path.inspect}"
    end

    def rc_block(path = bindir, fish: false)
      line = fish ? fish_path_line(path) : export_line(path)
      "#{MARKER_BEGIN}\n#{line}\n#{MARKER_END}"
    end

    def apply_rc(existing, path = bindir, fish: false)
      block = rc_block(path, fish: fish)
      if existing.include?(MARKER_BEGIN)
        existing.sub(
          /#{Regexp.escape(MARKER_BEGIN)}.*?#{Regexp.escape(MARKER_END)}/m,
          block
        )
      else
        suffix = existing.empty? || existing.end_with?("\n") ? "" : "\n"
        "#{existing}#{suffix}\n#{block}\n"
      end
    end

    # Profile/rc files to update so PATH works in login and interactive shells.
    # Returns [{ path:, fish: }, ...] in preference order (current $SHELL first).
    def shell_rc_entries(env = ENV)
      home = File.expand_path("~")
      shell = File.basename(env["SHELL"].to_s)
      paths = []

      case shell
      when "zsh"
        paths << File.join(home, ".zshrc")
        paths << File.join(home, ".zprofile") if darwin?
      when "bash"
        if darwin?
          paths << File.join(home, ".bash_profile")
        else
          paths << File.join(home, ".bashrc")
        end
      when "fish"
        paths << File.join(home, ".config", "fish", "config.fish")
      else
        paths << File.join(home, ".profile")
      end

      # Other profiles the user already has (e.g. bash + zsh on the same account).
      %w[
        .zshrc .zprofile .zshenv .bashrc .bash_profile .profile
        .config/fish/config.fish
      ].each do |rel|
        p = File.join(home, rel)
        paths << p if File.file?(p)
      end

      seen = {}
      paths.filter_map do |p|
        next if seen[p]
        seen[p] = true
        { path: p, fish: p.end_with?("config.fish") }
      end
    end

    # Primary rc for messages (first entry for $SHELL).
    def shell_rc_path(env = ENV)
      shell_rc_entries(env).first&.fetch(:path) || File.expand_path("~/.profile")
    end

    def skip_build?(env = ENV)
      truthy?(env["MVCC_SKIP_BUILD"])
    end

    def skip_path?(env = ENV)
      truthy?(env["MVCC_SKIP_PATH"]) || truthy?(env["CI"])
    end

    # MVCC_ASK_PATH=1 restores the interactive y/N prompt (default: write profiles automatically).
    def ask_path?(env = ENV)
      truthy?(env["MVCC_ASK_PATH"])
    end

    def run!(env: ENV, stdin: $stdin, stderr: $stderr)
      preflight!(env)
      build!(stderr) unless skip_build?(env)
      offer_path!(env: env, stdin: stdin, stderr: stderr)
    end

    # PATH gem install should use when resolving build tools (matches install_toolkit.sh).
    def effective_path(env = ENV)
      [
        File.expand_path("~/.cargo/bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
        env.fetch("PATH", "")
      ].join(File::PATH_SEPARATOR)
    end

    def find_executable(name, env = ENV)
      effective_path(env).split(File::PATH_SEPARATOR).each do |dir|
        next if dir.empty?

        path = File.join(dir, name)
        return path if File.executable?(path)
      end
      nil
    end

    # Returns human-readable errors; empty when ready to build.
    def preflight_errors(env = ENV)
      errors = []
      unless darwin?
        errors << "macOS required (gem install builds the Apple silicon toolchain)"
        return errors
      end
      unless arm64?
        errors << "Apple silicon (arm64) required; this host is #{host_cpu}"
        return errors
      end
      unless system("xcode-select", "-p", out: File::NULL, err: File::NULL)
        errors << "Xcode Command Line Tools (run: xcode-select --install)"
      end
      unless find_executable("brew", env)
        errors << "Homebrew (https://brew.sh)"
      end
      %w[cmake ninja cargo rustc clang clang++ ar ranlib].each do |cmd|
        errors << "#{cmd} (not on PATH)" unless find_executable(cmd, env)
      end
      if find_executable("brew", env)
        llvm = brew_llvm_prefix(env)
        if llvm.nil?
          errors << "Homebrew llvm (brew install llvm)"
        elsif !File.directory?(File.join(llvm, "lib", "cmake", "llvm"))
          errors << "Homebrew llvm CMake files at #{llvm}/lib/cmake/llvm (brew reinstall llvm)"
        end
      end
      errors
    end

    def preflight!(env = ENV)
      errors = preflight_errors(env)
      return if errors.empty?

      missing = errors.join("\n  - ")
      abort <<~MSG
        mvcc: install prerequisites missing:
          - #{missing}

        Install with:
          xcode-select --install
          brew install llvm cmake ninja
          curl -sSf https://sh.rustup.rs | sh

        Then open a new shell and re-run gem install.
      MSG
    end

    def build!(stderr = $stderr)
      script = File.join(gem_root, "tools", "install_toolkit.sh")
      abort "mvcc: #{script} is missing from the gem" unless File.file?(script)
      GemProgress.log "mvcc: building toolkit from source (tools/install_toolkit.sh). This takes several minutes."
      path = effective_path(ENV)
      ok = GemProgress.system(
        { "PATH" => path, "MVCC_INSTALL_FROM_GEM" => "1" },
        "bash",
        script
      )
      abort "mvcc: tools/install_toolkit.sh failed" unless ok
      abort "mvcc: toolkit nvcc missing after build (#{bindir})" unless File.executable?(File.join(bindir, "nvcc"))
    end

    def offer_path!(env: ENV, stdin: $stdin, stderr: $stderr)
      line = export_line
      entries = shell_rc_entries(env)
      GemProgress.log "mvcc: toolkit is at #{bindir}"
      GemProgress.log "mvcc: add this to PATH (gem-install equivalent of export PATH=\"$PWD/toolkit/bin:$PATH\"):"
      GemProgress.log "  #{line}"

      if skip_path?(env)
        GemProgress.log "mvcc: not editing shell profiles (CI or MVCC_SKIP_PATH). Add the line above yourself."
        return
      end

      if ask_path?(env)
        rc = shell_rc_path(env)
        unless stdin.tty?
          GemProgress.log "mvcc: stdin is not a terminal; not editing profiles."
          return
        end
        stderr.print "Add mvcc to PATH in your shell profile(s)? [Y/n] "
        stderr.flush
        answer = stdin.gets
        if answer && answer.strip.match?(/\An(o)?\z/i)
          GemProgress.log "mvcc: skipped. You can add the line later."
          return
        end
      end

      updated = []
      entries.each do |entry|
        rc = entry[:path]
        fish = entry[:fish]
        body = File.file?(rc) ? File.read(rc) : ""
        FileUtils.mkdir_p(File.dirname(rc)) if fish && !File.directory?(File.dirname(rc))
        new_body = apply_rc(body, bindir, fish: fish)
        next if new_body == body

        File.write(rc, new_body)
        updated << rc
      end

      if updated.empty?
        GemProgress.log "mvcc: shell profiles already include this toolkit on PATH."
      else
        GemProgress.log "mvcc: wrote PATH to:"
        updated.each { |rc| GemProgress.log "  #{rc}" }
        GemProgress.log "mvcc: open a new shell or source the file(s) above."
      end
    end

    def darwin?
      RbConfig::CONFIG["host_os"].to_s.include?("darwin")
    end

    def arm64?
      host_cpu.match?(/\A(arm64|aarch64)\z/i)
    end

    def host_cpu
      RbConfig::CONFIG["host_cpu"].to_s
    end

    def brew_llvm_prefix(env = ENV)
      brew = find_executable("brew", env)
      return nil unless brew

      prefix = `#{brew} --prefix llvm 2>/dev/null`.strip
      return nil if prefix.empty? || !File.directory?(prefix)

      prefix
    end

    def truthy?(value)
      %w[1 true yes].include?(value.to_s.downcase)
    end
  end
end
