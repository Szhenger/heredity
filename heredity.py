import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Initialize joint probability to 1
    probability = 1
    for person in people:

        # Find parents of person if exists
        mother = people[person]["mother"]
        father = people[person]["father"]

        # Get probabilitiy distrubutions for parents
        mother_probabilities = {
            mother in one_gene: 0.5,
            mother in two_genes: 1 - PROBS["mutation"]
        }
        mother_probability = mother_probabilities.get(True, PROBS["mutation"])
        father_probabilities = {
            father in one_gene: 0.5,
            father in two_genes: 1 - PROBS["mutation"]
        }
        father_probability = father_probabilities.get(True, PROBS["mutation"])

        if person in one_gene:

            # Initialize probability for one gene
            one_probability = 1

            # Unconditional probability with no parents
            if mother is None and father is None:
                one_probability *= PROBS["gene"][1]

            # Conditional probability with parents
            else:
                one_probability *= mother_probability * (1 - father_probability) + father_probability * (1 - mother_probability)

            if person in have_trait:
                one_probability *= PROBS["trait"][1][True]
            else:
                one_probability *= PROBS["trait"][1][False]

            # Multiply to joint probability
            probability *= one_probability

        elif person in two_genes:

            # Initilaize probability for two genes
            two_probability = 1

            # Unconditional probability with no parents
            if mother is None and father is None:
                two_probability *= PROBS["gene"][2]

            # Conditional probability with parents
            else:
                two_probability *= mother_probability * father_probability

            if person in have_trait:
                two_probability *= PROBS["trait"][2][True]
            else:
                two_probabilty *= PROBS["trait"][2][False]

            # Multiply to joint probability
            probability *= two_probability
        else:

            # Initialize probability with no genes
            zero_probability = 1

            # Unconditional probability with no parents
            if mother is None and father is None:
                zero_probability *= PROBS["gene"][0]

            # Conditional probability with parents
            else:
                zero_probability *= (1 - mother_probability) * (1 - father_probability)

            if person in have_trait:
                zero_probability *= PROBS["trait"][0][True]
            else:
                zero_probability *= PROBS["trait"][0][False]

            # Multiply to joint probability
            probability *= zero_probability

    # Return joint probability
    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:

        # Add p to corresponding genes per person
        genes = {
            person in one_gene: 1,
            person in two_genes: 2
        }
        gene = genes.get(True, 0)
        probabilities[person]["gene"][gene] += p

        # Add p to corresponding traits per person
        traits = {
            person in have_trait: True,
            person not in have_trait: False
        }
        trait = traits.get(True, None)
        probabilities[person]["trait"][trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:

        # Normalize gene values per person
        gene_sum = 0
        for i in range(3):
            gene_sum += probabilities[person]["gene"][i]
        for i in range(3):
            probabilities[person]["gene"][i] /= gene_sum

        # Normalize trait values per person
        trait_sum = 0
        for bool in (True, False):
            trait_sum += probabilities[person]["trait"][bool]
        for bool in (True, False):
            probabilities[person]["trait"][bool] /= trait_sum


if __name__ == "__main__":
    main()
