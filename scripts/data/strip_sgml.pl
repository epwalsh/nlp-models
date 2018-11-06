#!/usr/bin/env perl
#
use strict;
use warnings;

use HTML::Parser;

my $file = $ARGV[0];

HTML::Parser->new(default_h => [""],
    text_h => [ sub { print shift }, 'text' ]
  )->parse_file($file) or die "Failed to parse $file: $!";

# https://stackoverflow.com/questions/36827515/convert-sgm-to-txt
