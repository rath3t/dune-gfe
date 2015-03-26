#!/usr/bin/env bash

GIT_CLEANUP=$PWD/git-cleanup.bash
GIT_CONVERT=$PWD/convert-to-git.bash

if ! [[ -f ${GIT_CONVERT} ]]; then
    cat <<EOF
error: This script expects to be called convert-to-git.bash and reside in the
       directory in which it is executed.
EOF
    exit -1
fi

clone() {
    git svn clone --stdlayout \
        https://www.igpm.rwth-aachen.de/svn/sander/dune-modules/$1
}

handle_gitignore_and_filter() {
    git svn show-ignore > .gitignore
    git add .gitignore
    git commit -m 'Convert svn:ignore properties to .gitignore.'

    git filter-branch -f \
        --msg-filter '
            sed -e "/^\$/{ :l; N; s/^\n\$//; t l; p; d; }" | \
            sed -e "/^git-svn-id:/s/.*@\([^ ]*\) .*$/[[Imported from SVN: r\1]]/"' \
        --commit-filter ". ${GIT_CLEANUP};" \
        HEAD
}

create_release_branch() {
    git checkout -b releases/$1 origin/$2
}

create_git_cleanup() {
    cat > ${GIT_CLEANUP} <<'EOF'
#!/usr/bin/env bash

case $GIT_AUTHOR_NAME in
    akbib*|youett*)
        GIT_AUTHOR_NAME="Jonathan Youett"
        GIT_AUTHOR_EMAIL=youett@mi.fu-berlin.de
        ;;
    ansgar)
        GIT_AUTHOR_NAME="Ansgar Burchardt"
        GIT_AUTHOR_EMAIL=burchardt@igpm.rwth-aachen.de
        ;;
    forster*)
        GIT_AUTHOR_NAME="Ralf Forster"
        GIT_AUTHOR_EMAIL=forster@math.fu-berlin.de
        ;;
    graeser*)
        GIT_AUTHOR_NAME="Carsten Gräser"
        GIT_AUTHOR_EMAIL=graeser@mi.fu-berlin.de
        ;;
    lschmidt*|oel*)
        GIT_AUTHOR_NAME="Leo Schmidt"
        GIT_AUTHOR_EMAIL=lschmidt@math.fu-berlin.de
        ;;
    klapprot*)
        GIT_AUTHOR_NAME="Corinna Klapproth"
        GIT_AUTHOR_EMAIL=klapproth@zib.de
        ;;
    mawolf*)
        GIT_AUTHOR_NAME="Maren-Wanda Wolf"
        GIT_AUTHOR_EMAIL=mawolf@math.fu-berlin.de
        ;;
    maxka)
        GIT_AUTHOR_NAME="Max Kahnt"
        GIT_AUTHOR_EMAIL=max.kahnt@fu-berlin.de
        ;;
    pipping*)
        GIT_AUTHOR_NAME="Elias Pipping"
        GIT_AUTHOR_EMAIL=elias.pipping@fu-berlin.de
        ;;
    podlesjo)
        GIT_AUTHOR_NAME="Joscha Podlesny"
        GIT_AUTHOR_EMAIL=joscha.py@googlemail.com
        ;;
    sander*)
        GIT_AUTHOR_NAME="Oliver Sander"
        GIT_AUTHOR_EMAIL=sander@igpm.rwth-aachen.de
        ;;
    sertel)
        GIT_AUTHOR_NAME="Susanne Ertel"
        GIT_AUTHOR_EMAIL=ertel@zib.de
        ;;
    usack*)
        GIT_AUTHOR_NAME="Uli Sack"
        GIT_AUTHOR_EMAIL=usack@math.fu-berlin.de
        ;;
esac
git commit-tree "$@"
EOF
}

save_script_and_create_tag() {
    cp ${GIT_CONVERT} .
    git add convert-to-git.bash
    git commit -m 'Document subversion->git migration process'
    git tag -a \
        -m "Subversion->Git migration completed. Master rev: $@" \
        'subversion->git'
}

### Check out and clean up repositories

create_git_cleanup

clone dune-gfe
(
    cd dune-gfe
    master_rev=$(git svn find-rev HEAD)
    create_release_branch 2.0-1 2.0-1 ; handle_gitignore_and_filter; # Rename
    create_release_branch 2.1-1 2.1-1   ; handle_gitignore_and_filter;
    create_release_branch 2.2-1 2.2-1   ; handle_gitignore_and_filter;
    create_release_branch 2.2-2 2.2-2   ; handle_gitignore_and_filter;
    create_release_branch 2.3-1 2.3-1   ; handle_gitignore_and_filter;
    create_release_branch 2.3-2 2.3-2   ; handle_gitignore_and_filter;
    create_release_branch 2.3-last-basically-sequential-version  2.3-last-basically-sequential-version                     ; handle_gitignore_and_filter;
    create_release_branch first-evidence-that-neff-shells-do-not-lock first-evidence-that-neff-shells-do-not-lock          ; handle_gitignore_and_filter;
    create_release_branch first-working-adol-c-support first-working-adol-c-support                                        ; handle_gitignore_and_filter;
    create_release_branch working-version-before-moving-to-dune-functions working-version-before-moving-to-dune-functions  ; handle_gitignore_and_filter;
    git checkout master                 ; handle_gitignore_and_filter;
    save_script_and_create_tag ${master_rev}
)


rm -f ${GIT_CLEANUP}

## Debugging section
#
# Use the following scripts to check if the conversion went according to plan
#
# (1) To generate a list of authors
#
#       !/usr/bin/env bash
#
#       for i in dune-*; do
#           (
#               cd $i;
#               git log --format=tformat:'%an <%ae>';
#           );
#       done | perl -e '
#           use 5.018;
#           my %a;
#           while (<>) {
#               chomp; $a{$_}++;
#           };
#           while (each %a) {
#               printf "%d\t%s\n", $a{$_}, $_;
#           }
#       ' | sort -n
#
# (2) To generate a list of converted branches
#
#       !/usr/bin/env bash
#
#       for i in dune-*; do
#           (
#               cd $i;
#               echo $i;
#               git for-each-ref --format='  %(refname)' refs/heads;
#           );
#       done
