# frozen_string_literal: true

require "shellwords"

module Mvcc
  # RubyGems hides extconf stdout/stderr during `gem install`. Log and stream
  # child build output on /dev/tty when the user has a terminal.
  module GemProgress
    module_function

    def io
      return @io if defined?(@io) && @io

      @io = File.open("/dev/tty", "w")
    rescue StandardError
      @io = $stderr
    end

    def tty?
      return @tty unless @tty.nil?

      File.open("/dev/tty", "w").close
      @tty = true
    rescue StandardError
      @tty = false
    end

    def log(message)
      io.puts message
      io.flush
    end

    def system(env, *argv)
      env = env.to_h.transform_keys(&:to_s)
      if tty?
        cmd = Shellwords.shelljoin(argv)
        exports = env.map { |k, v| "export #{Shellwords.escape(k)}=#{Shellwords.escape(v)}" }.join("; ")
        prelude = exports.empty? ? "" : "#{exports}; "
        Kernel.system("bash", "-c", "#{prelude}exec >/dev/tty 2>&1; #{cmd}")
      else
        Kernel.system(env, *argv)
      end
    end
  end
end
