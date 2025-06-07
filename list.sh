#!/usr/bin/fish

# Get directories sorted by modification time, extract relevant fields
for line in (ls -l --time=mtime --sort=time | grep '^d' | awk '{print $6, $7, $8, $9}')
    # Split the line into date, time, timezone, and directory name
    set -l parts (string split " " $line)
    set -l date $parts[1]
    set -l time $parts[2]
    set -l zone $parts[3]
    set -l dir $parts[4]

    # Calculate directory size, suppress permission errors
    set -l size (du -sh "$dir" 2>/dev/null | cut -f1)

    # Output in the format: size date time zone directory
    echo "$size $date $time $zone $dir"
end
