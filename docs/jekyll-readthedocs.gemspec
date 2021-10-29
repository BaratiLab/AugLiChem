# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "jekyll-readthedocs"
  spec.version       = "0.4.1"
  spec.authors       = ["Timothée Mazzucotelli"]
  spec.email         = ["timothee.mazzucotelli@pm.me"]

  spec.summary       = "A ReadTheDocs theme based on minima."
  spec.homepage      = "https://github.com/pawamoy/jekyll-readthedocs"
  spec.license       = "MIT"

  spec.metadata["plugin_type"] = "theme"

  spec.files = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r!^(assets|_(includes|layouts|sass)/|(LICENSE|README)((\.(txt|md|markdown)|$)))!i)
  end

  spec.add_runtime_dependency "jekyll", "~> 3.5"
  spec.add_runtime_dependency "jekyll-feed", "~> 0.11"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.5"
  spec.add_development_dependency "bundler", "~> 1.15"
end
