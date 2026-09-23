#!/usr/bin/env ruby
# frozen_string_literal: true

# Path-prompt helpers for gem install. Does not run install_toolkit.sh.
# Run: ruby tests/gem/installer.rb

require_relative "../../lib/mvcc"
require_relative "../../lib/mvcc/installer"

fail = 0
def check(name, cond)
  if cond
    puts "  [ok]   #{name}"
  else
    $fail += 1
    puts "  [FAIL] #{name}"
  end
end
$fail = 0

root = Mvcc::Installer.gem_root
check("gem root is the repo", File.file?(File.join(root, "tools/install_toolkit.sh")))
check("Mvcc.bindir is toolkit/bin", Mvcc.bindir.end_with?("/toolkit/bin"))
check("export uses absolute toolkit/bin, not $PWD",
      Mvcc::Installer.export_line.include?(Mvcc::Installer.bindir) &&
        !Mvcc::Installer.export_line.include?("$PWD"))

block = Mvcc::Installer.rc_block("/tmp/mvcc-toolkit/bin")
check("rc block has markers",
      block.include?(Mvcc::Installer::MARKER_BEGIN) &&
        block.include?(Mvcc::Installer::MARKER_END))
check("rc block has the export",
      block.include?('export PATH="/tmp/mvcc-toolkit/bin:$PATH"'))

updated = Mvcc::Installer.apply_rc("export PATH=/usr/bin\n", "/opt/mvcc/bin")
check("apply_rc appends a new block",
      updated.include?("export PATH=/usr/bin") &&
        updated.include?('export PATH="/opt/mvcc/bin:$PATH"'))

replaced = Mvcc::Installer.apply_rc(updated, "/opt/mvcc-2/bin")
check("apply_rc replaces an existing block",
      replaced.include?('export PATH="/opt/mvcc-2/bin:$PATH"') &&
        !replaced.include?('export PATH="/opt/mvcc/bin:$PATH"'))

check("skip_path? on CI", Mvcc::Installer.skip_path?({ "CI" => "true" }))
check("skip_path? on MVCC_SKIP_PATH", Mvcc::Installer.skip_path?({ "MVCC_SKIP_PATH" => "1" }))
check("skip_build? on MVCC_SKIP_BUILD", Mvcc::Installer.skip_build?({ "MVCC_SKIP_BUILD" => "1" }))
check("ask_path? on MVCC_ASK_PATH", Mvcc::Installer.ask_path?({ "MVCC_ASK_PATH" => "1" }))
check("shell_rc_entries includes zshrc for zsh",
      Mvcc::Installer.shell_rc_entries({ "SHELL" => "/bin/zsh" }).any? { |e| e[:path].end_with?(".zshrc") })
check("fish rc block uses fish_add_path",
      Mvcc::Installer.rc_block("/tmp/bin", fish: true).include?("fish_add_path"))

check("extconf loads the installer",
      File.read(File.join(root, "ext/mvcc/extconf.rb")).include?("Mvcc::Installer.run!"))
check("lib/mvcc.rb loads cuda sugar",
      File.read(File.join(root, "lib/mvcc.rb")).include?('require_relative "mvcc/cuda"'))
check("gem install logs via /dev/tty",
      File.file?(File.join(root, "lib/mvcc/gem_progress.rb")) &&
        File.read(File.join(root, "lib/mvcc/installer.rb")).include?("GemProgress.log"))
script = File.read(File.join(root, "tools/install_toolkit.sh"))
check("install_toolkit.sh does not hide install_name_tool failure under set -e",
      script.include?("install_name_tool") &&
        !script.match?(/install_name_tool[^\n]*2>\/dev\/null/) &&
        script.include?("leaving @rpath"))
check("libcudart is linked with headerpad so gem paths fit",
      File.read(File.join(root, "crates/mvcc-cudart/build.rs"))
          .include?("headerpad_max_install_names"))
check("install_toolkit.sh survives a missing controlling tty",
      begin
        require "open3"
        require "tempfile"
        script = File.join(root, "tools/install_toolkit.sh")
        # Only the tty prelude runs before the first cmake; stop the script right after it.
        probe = File.read(script).sub(/^run_cmake\(\) \{.*/m, "exit 0\n")
        Tempfile.create(["install_toolkit_probe", ".sh"]) do |f|
          f.write(probe)
          f.flush
          out, status = Open3.capture2e({ "MVCC_INSTALL_FROM_GEM" => "1" }, "bash", f.path, stdin_data: "")
          status.success? && !out.include?("Device not configured")
        end
      rescue StandardError
        false
      end)
check("extconf does not call gem push",
      !File.read(File.join(root, "ext/mvcc/extconf.rb")).match?(/^\s*gem push\b/))

check("effective_path includes ~/.cargo/bin",
      Mvcc::Installer.effective_path.include?(File.expand_path("~/.cargo/bin")))
check("find_executable finds bash", !Mvcc::Installer.find_executable("bash").to_s.empty?)
check("preflight_errors is an Array", Mvcc::Installer.preflight_errors.is_a?(Array))
check("preflight! calls preflight_errors",
      File.read(File.join(root, "lib/mvcc/installer.rb")).include?("preflight_errors"))

exit($fail.zero? ? 0 : 1)
